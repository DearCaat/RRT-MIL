import torch
import torch.nn as nn
import torch.nn.functional as F


class DAttention(nn.Module):
    def __init__(self, n_classes, dropout, act, n_features=1024):
        super(DAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(n_features, 512)]

        if act.lower() == "gelu":
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K))
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.squeeze()
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        logits = self.classifier(M)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S
