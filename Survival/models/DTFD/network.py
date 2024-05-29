import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
import sys
import random
import numpy as np

from .rrt import RRTEncoder


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum("bgf,cf->bcg", [features, tweight])
    return cam_maps


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0, n_robust=0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

        self.apply(initialize_weights)

        if n_robust > 0:
            # 改变init
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)
            # 改变后续的随机状态
            [torch.rand((1024, 512)) for i in range(n_robust)]

    def forward(self, x):
        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, dropout=False, act="relu", n_robust=0, rrt=False, rrt_convk=15, rrt_moek=3, rrt_as=False, rrt_md=False):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        # 添加自己的re-embedding模块
        self.rrt = RRTEncoder(attn="rrt", pool="none", n_layers=2, epeg_k=rrt_convk, crmsa_k=rrt_moek, moe_mask_diag=rrt_md, init=True, rrt_window_num=16) if rrt else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True) if act.lower() == "relu" else nn.GELU()
        self.drop = nn.Dropout(0.25)
        self.dropout = dropout
        self.apply(initialize_weights)

        if n_robust > 0:
            # 改变init
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)
            # 改变后续的随机状态
            [torch.rand((1024, 512)) for i in range(n_robust)]

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        if self.dropout:
            x = self.drop(x)
        if isinstance(self.rrt, RRTEncoder):
            x, _ = self.rrt(x)
        else:
            x = self.rrt(x)
        return x


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0, n_robust=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention(L, D, K, n_robust=n_robust)
        self.classifier = Classifier_1fc(L, num_cls, droprate, n_robust=n_robust)

    def forward(self, x):  # x: N x L
        AA = self.attention(x)  # K x N
        afeat = torch.mm(AA, x)  # K x L
        pred = self.classifier(afeat)  # K x num_cls
        return pred


class Attention(nn.Module):
    def __init__(self, L=512, D=128, K=1, n_robust=0):
        super(Attention, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())

        self.attention_weights = nn.Linear(self.D, self.K)

        self.apply(initialize_weights)

        if n_robust > 0:
            # 改变init
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)
            # 改变后续的随机状态
            [torch.rand((1024, 512)) for i in range(n_robust)]

    def forward(self, x, isNorm=True):
        # x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  # K x N


class DTFD(nn.Module):
    def __init__(self, lr, weight_decay, steps, criterion, rrt=False, epeg_k=15, crmsa_k=3, input_dim=1024, inner_dim=512, n_classes=2, group=8, distill="MaxMinS") -> None:
        super().__init__()
        self.classifier = Classifier_1fc(inner_dim, n_classes, 0.25)
        self.attention = Attention(inner_dim)
        self.dimReduction = DimReduction(input_dim, inner_dim, rrt=rrt, dropout=0.25, rrt_convk=epeg_k, rrt_moek=crmsa_k)
        self.UClassifier = Attention_with_Classifier(L=inner_dim, num_cls=n_classes, droprate=0.25)
        self.group = group
        self.distill = distill
        self.criterion = criterion
        trainable_parameters = []
        trainable_parameters += list(self.classifier.parameters())
        trainable_parameters += list(self.attention.parameters())
        trainable_parameters += list(self.dimReduction.parameters())
        # optimizer
        self.optimizer0 = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=weight_decay)
        self.scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer0, steps, 0)

    def train_forward(self, x, label):
        feat_index = list(range(x.shape[0]))

        index_chunk_list = np.array_split(np.array(feat_index), self.group)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]
        slide_pseudo_feat = []
        # pseudo-bag prediction
        slide_sub_hazards = []
        slide_sub_survival = []
        # ground-truth
        slide_sub_labels = []
        slide_sub_censorship = []
        for tindex in index_chunk_list:
            slide_sub_labels.append(label["label"])
            slide_sub_censorship.append(label["censorship"])
            subFeat_tensor = torch.index_select(x, dim=0, index=torch.LongTensor(tindex).to(x.device))
            tmidFeat = self.dimReduction(subFeat_tensor)
            tAA = self.attention(tmidFeat).squeeze(0)

            tattFeats = torch.einsum("ns,n->ns", tmidFeat, tAA)  # n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  # 1 x fs
            tPredict = self.classifier(tattFeat_tensor)  # 1 x 2
            # sruvival-related prediction
            tHazards = torch.sigmoid(tPredict)
            tS = torch.cumprod(1 - tHazards, dim=1)
            slide_sub_hazards.append(tHazards)
            slide_sub_survival.append(tS)
            #
            patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  # cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  # n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  # n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            topk_idx_max = sort_idx[:1].long()
            topk_idx_min = sort_idx[-1:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if self.distill == "MaxMinS":
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif self.distill == "MaxS":
                slide_pseudo_feat.append(max_inst_feat)
            elif self.distill == "AFS":
                slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        # pseudo-bag loss
        slide_sub_hazards = torch.cat(slide_sub_hazards, dim=0)  # numGroup x fs
        slide_sub_survival = torch.cat(slide_sub_survival, dim=0)  # numGroup x fs
        # backward
        loss0 = self.criterion(hazards=slide_sub_hazards, S=slide_sub_survival, Y=torch.tensor(slide_sub_labels).to(x.device), c=torch.tensor(slide_sub_censorship).to(x.device))
        self.optimizer0.zero_grad()
        loss0.backward(retain_graph=True)

        logits = self.UClassifier(slide_pseudo_feat)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S

    def test_forward(self, x):
        tfeat = x
        midFeat = self.dimReduction(tfeat)
        AA = self.attention(midFeat, isNorm=False).squeeze(0)  # N

        feat_index = list(range(tfeat.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), self.group)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        slide_d_feat = []

        for tindex in index_chunk_list:
            idx_tensor = torch.LongTensor(tindex).to(x.device)
            tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

            tAA = AA.index_select(dim=0, index=idx_tensor)
            tAA = torch.softmax(tAA, dim=0)
            tattFeats = torch.einsum("ns,n->ns", tmidFeat, tAA)  # n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  # 1 x fs

            patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  # cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  # n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  # n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

            if self.distill == "MaxMinS":
                topk_idx_max = sort_idx[:1].long()
                topk_idx_min = sort_idx[-1:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                slide_d_feat.append(d_inst_feat)
            elif self.distill == "MaxS":
                topk_idx_max = sort_idx[:1].long()
                topk_idx = topk_idx_max
                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                slide_d_feat.append(d_inst_feat)
            elif self.distill == "AFS":
                slide_d_feat.append(tattFeat_tensor)

        slide_d_feat = torch.cat(slide_d_feat, dim=0)
        logits = self.UClassifier(slide_d_feat)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S

    def forward(self, x, label=None):
        x = x.squeeze(0)

        # print('In DTFD, x.shape: ', x.shape)
        if self.training:
            return self.train_forward(x, label)
        else:
            return self.test_forward(x)


if __name__ == "__main__":
    x = torch.rand(100, 1024)
    y = torch.ones(
        1,
    ).long()
    # group=8,distill='MaxMinS' 这是它在LUAD&LUSC上的默认参数，原论文给的
    # 看优化的部分需不需要拿出去
    dtfd = DTFD(
        x.device,
        1e-5,
        1e-5,
        100,
    )
    # 训练需要label，它要在内部算损失,然后直接在内部就把损失后向传播了
    pred = dtfd(x, y)
    print(pred.size())  # 1,C
    dtfd.eval()
    x_test = dtfd(x)
    print(x_test.size())  # 1,C
