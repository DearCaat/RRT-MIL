import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MeanMIL(nn.Module):
    def __init__(self, n_features=1024, n_classes=1, dropout=True, act='relu'):
        super(MeanMIL, self).__init__()

        head = [nn.Linear(n_features, 512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]

        head += [nn.Linear(512, n_classes)]

        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def forward(self, x):
        logits = self.head(x).mean(axis=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S


class MaxMIL(nn.Module):
    def __init__(self, n_features=1024, n_classes=1, dropout=True, act='relu'):
        super(MaxMIL, self).__init__()

        head = [nn.Linear(n_features, 512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
        head += [nn.Linear(512, n_classes)]
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def forward(self, x):
        logits, _ = self.head(x).max(axis=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S
