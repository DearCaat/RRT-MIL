"""
Shao Z, Bian H, Chen Y, et al. Transmil: Transformer based correlated multiple instance learning for whole slide image classification[J]. Advances in neural information processing systems, 2021, 34: 2136-2147.
"""

import numpy as np
import torch
import torch.nn as nn
from .util import NystromAttention


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            pinv_iterations=6,
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
        x = self.proj(cnn_feat) + cnn_feat + \
            self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, dropout, act, n_features=1024):
        super(TransMIL, self).__init__()
        ###
        self._fc1 = [nn.Linear(n_features, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(0.25)]
        self._fc1 = nn.Sequential(*self._fc1)

        self.pos_layer = PPEG(dim=512)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Linear(512, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, x):
        h = x.float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 256]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 256]

        # ---->cls_token
        cls_tokens = self.cls_token.expand(1, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 256]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 256]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 256]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self.classifier(h)  # [B, n_classes]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S


class TransMIL_NO_PPEG(nn.Module):
    def __init__(self, n_classes, dropout, act, n_features=1024):
        super(TransMIL_NO_PPEG, self).__init__()
        ###
        self._fc1 = [nn.Linear(n_features, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(0.25)]
        self._fc1 = nn.Sequential(*self._fc1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Linear(512, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, x):
        h = x.float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 256]

        # # ---->pad
        # H = h.shape[1]
        # _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        # add_length = _H * _W - H
        # h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 256]

        # ---->cls_token
        cls_tokens = self.cls_token.expand(1, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 256]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 256]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self.classifier(h)  # [B, n_classes]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S
