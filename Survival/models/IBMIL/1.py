import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .util import NystromAttention
import os

# official code of transmil


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, input_size, confounder_path=None, k="all"):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)
        self.confounder_path = confounder_path
        if confounder_path:
            print("load confounder from: ", confounder_path)
            self.confounder_path = confounder_path
            conf_list = []
            for root, dirs, files in os.walk(confounder_path):
                for file in files:
                    if file.endswith(".npy"):
                        print(os.path.join(root, file))
                        conf_list.append(torch.from_numpy(np.load(os.path.join(root, file))).float())
            if k == "all":
                conf_tensor = torch.cat(conf_list, 0)
            else:
                conf_tensor = conf_list[k]
            self.register_buffer("confounder_feat", conf_tensor)
            joint_space_dim = 128
            dropout_v = 0.1
            self.confounder_W_q = nn.Linear(512, joint_space_dim)
            self.confounder_W_k = nn.Linear(512, joint_space_dim)
            self._fc2 = nn.Linear(1024, self.n_classes)
            self.norm2 = nn.LayerNorm(1024)

    def forward(self, feats):
        ## h = kwargs['data'].float() #[B, n, 1024]

        h = feats
        # print(h.shape)

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h_not_norm = h[:, 0]
        A = None
        if self.confounder_path:
            norm = False
            if "nn" not in self.confounder_path[0]:  # un normalized
                h = self.norm(h)[:, 0]
                device = h.device
                bag_q = self.confounder_W_q(h)
                conf_k = self.confounder_W_k(self.confounder_feat)
                A = torch.mm(conf_k, bag_q.transpose(0, 1))
                A = F.softmax(A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0)  # normalize attention scores, A in shape N x C,
                conf_feats = torch.mm(A.transpose(0, 1), self.confounder_feat)  # compute bag representation, B in shape C x V
                h = torch.cat((h, conf_feats), dim=1)
                # print('not norm')

            else:
                device = h_not_norm.device
                bag_q = self.confounder_W_q(h_not_norm)
                conf_k = self.confounder_W_k(self.confounder_feat)
                A = torch.mm(conf_k, bag_q.transpose(0, 1))
                A = F.softmax(A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0)  # normalize attention scores, A in shape N x C,
                conf_feats = torch.mm(A.transpose(0, 1), self.confounder_feat)  # compute bag representation, B in shape C x V
                h = torch.cat((h, conf_feats.unsqueeze(1).repeat(1, h.shape[1], 1)), dim=-1)
                h = self.norm2(h)[:, 0]
                # print(' norm')
        else:
            h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        return logits


if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data=data)
    print(results_dict)
