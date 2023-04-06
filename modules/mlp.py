from torch import nn
class MlpHead(nn.Module):

    def __init__(self, mlp_dim = 512, hid_dim=512, out_dim=1, bn=False):
        super().__init__()
        self.out_dim = out_dim
        self.fc1 = nn.Linear(mlp_dim, hid_dim)
        self.bn = nn.LazyBatchNorm1d() if bn else None
        # c16上，relu换成gelu会降低性能 nn.GELU()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, shape[-1])
        x = self.fc1(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.reshape(*shape[:-1], self.out_dim)
        return x
class MlpHeadDINO(nn.Module):
   
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=2, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            layers = [nn.Linear(in_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        # self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x