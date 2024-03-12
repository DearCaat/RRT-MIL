import torch.nn as nn

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # ref from meituan
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MeanMIL(nn.Module):
    def __init__(self,input_dim,n_classes=1,dropout=True,act='relu',rrt=None):
        super(MeanMIL, self).__init__()

        head = [nn.Linear(input_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
        
        if rrt is not None:
            head += [rrt]

        head += [nn.Linear(512,n_classes)]
        
        self.head = nn.Sequential(*head)


        self.apply(initialize_weights)

    def forward(self,x):

        x = self.head(x).mean(axis=1)
        return x



class MaxMIL(nn.Module):
    def __init__(self,input_dim,n_classes=1,dropout=True,act='relu',rrt=None):
        super(MaxMIL, self).__init__()

        head = [nn.Linear(1024,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
            
        if rrt is not None:
            head += [rrt]
        head += [nn.Linear(512,n_classes)]
        self.head = nn.Sequential(*head)

        self.apply(initialize_weights)

    def forward(self,x):
        x,_ = self.head(x).max(axis=1)
        return x