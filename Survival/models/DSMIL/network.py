import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        # ref from https://github.com/Meituan-AutoML/Twins/blob/4700293a2d0a91826ab357fc5b9bc1468ae0e987/gvt.py#L356
        # if isinstance(m, nn.Conv2d):
        #     fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     fan_out //= m.groups
        #     m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #     if m.bias is not None:
        #         m.bias.data.zero_()
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1, dropout=True, act="relu", input_dim=1024):
        super(FCLayer, self).__init__()
        # self.embed = nn.Sequential(nn.Linear(1024, 512),nn.ReLU())
        # self.embed = [nn.Linear(1024, 512), nn.ReLU()]

        self.embed = [nn.Linear(input_dim, 512)]

        if act.lower() == "gelu":
            self.embed += [nn.GELU()]
        else:
            self.embed += [nn.ReLU()]

        if dropout:
            self.embed += [nn.Dropout(0.25)]

        self.embed = nn.Sequential(*self.embed)
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        feats = self.embed(feats)
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=True):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(nn.Dropout(dropout_v), nn.Linear(input_size, input_size), nn.ReLU())
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class MILNet(nn.Module):
    def __init__(self, n_classes, dropout=0.0, act="relu", input_dim=1024, rrt=False, rrt_convk=15, rrt_moek=3, rrt_as=False, rrt_md=False, no_rrt_init=False, no_init=True):
        super(MILNet, self).__init__()

        self.patch_to_emb = [nn.Linear(input_dim, 512)]
        if act.lower() == "relu":
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == "gelu":
            self.patch_to_emb += [nn.GELU()]
        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        self.dp = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.rrt = RRTEncoder(attn="rrt", pool="none", n_layers=2, epeg_k=rrt_convk, crmsa_k=rrt_moek, all_shortcut=rrt_as, moe_mask_diag=rrt_md, init=not no_rrt_init) if rrt else nn.Identity()

        # self.i_classifier = FCLayer(512,n_classes,dropout,act,input_dim=input_dim)
        self.i_classifier = nn.Linear(512, n_classes)
        self.b_classifier = BClassifier(512, n_classes)

        if not no_init:
            self.apply(initialize_weights)

    def forward(self, x, label=None, loss=None):
        ps = x.size(1)

        x = self.patch_to_emb(x)
        feats = self.dp(x.squeeze())

        prediction_ins = self.i_classifier(self.rrt(feats))
        prediction_bag, A, B = self.b_classifier(feats, prediction_ins)
        return prediction_ins, prediction_bag
