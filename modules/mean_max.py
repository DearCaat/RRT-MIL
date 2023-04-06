import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .swin import SwinEncoder
#from .setmil import ASPP
# import sys
# sys.stdout = open('data1.log', mode='w', encoding='utf-8')

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
    def __init__(self,n_classes=1,dropout=True,act='relu',test=False):
        super(MeanMIL, self).__init__()

        head = [nn.Linear(192,192)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
            
        #head += [SwinEncoder(attn='swin',pool='none')]
        #head += [ASPP(28,512,embed_dim=128)]
        head += [nn.Linear(192,n_classes)]
        
        self.head = nn.Sequential(*head)


        # self.head = nn.Sequential(
        #     nn.Linear(1024,512),
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(512,n_classes)
        # )

        if test:
            self._test = nn.Linear(1024, 512)
        # if init:
        #     pre_dict = torch.load(init)
        #     new_state_dict ={}
        #     target = ['head.0.weight','head.0.bias']
        #     for k,v in pre_dict.items():
        #         if k in target:
        #             new_state_dict[k.split('.',1)[1]]=v
        #             print(k)
        #     self.head.load_state_dict(new_state_dict,strict=False)
        #     print('embedding fc Inited')
        # else:
        self.apply(initialize_weights)

    def forward(self,x):

        x = self.head(x).mean(axis=1)
        return x



class MaxMIL(nn.Module):
    def __init__(self,n_classes=1,dropout=True,act='relu',test=False):
        super(MaxMIL, self).__init__()

        #head = [nn.Linear(192,192)]

        head = [nn.Linear(1024,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
        head += [SwinEncoder(attn='swin',pool='none',trans_conv=True)]
        #head += [nn.Linear(192,n_classes)]
        head += [nn.Linear(512,n_classes)]
        self.head = nn.Sequential(*head)

        if test:
            self._test = nn.Linear(1024, 512)
        # if init:
        #     pre_dict = torch.load(init)
        #     new_state_dict ={}
        #     target = ['head.0.weight','head.0.bias']
        #     for k,v in pre_dict.items():
        #         if k in target:
        #             new_state_dict[k.split('.',1)[1]]=v
        #     self.head.load_state_dict(new_state_dict,strict=False)
        #     print('embedding fc Inited')
        # else:
        self.apply(initialize_weights)

    def forward(self,x):
        x,_ = self.head(x).max(axis=1)
        return x

class FCLayer(nn.Module):
    def __init__(self, dropout=True,act='relu'):
        super(FCLayer, self).__init__()
        self.embed = [nn.Linear(1024, 512)]
        self.embed.append(SwinEncoder(attn='swin',pool='none'))
        # self.embed = nn.ModuleList([nn.Linear(1024, 512)])
        
        if act.lower() == 'gelu':
            self.embed += [nn.GELU()]
        else:
            self.embed += [nn.ReLU()]

        if dropout:
            self.embed += [nn.Dropout(0.25)]

        self.embed = nn.Sequential(*self.embed)

    def forward(self, feats):
        feats = self.embed(feats)
        return feats

class DAttention(nn.Module):
    def __init__(self,out_dim=2,n_robust=0):
        super(DAttention, self).__init__()
        self.embedding = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        
        self.head = nn.Linear(512,out_dim)

    def forward(self, x):
        # b,p,n = x.size()
        x = self.embedding(x) # 1024->512
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        x = self.head(x.squeeze(1))

        return x
class Dattention_ori(nn.Module):
    def __init__(self,out_dim=2):
        super(Dattention_ori,self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.embedding = FCLayer()
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.head = nn.Linear(512,out_dim)

    def forward(self,x):
        x=x.squeeze()
        x = self.embedding(x) # 1024->512
        A = self.attention(x)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        x = torch.mm(A, x)  # KxL
        x = self.head(x)
        return x
class Dattention_gated(nn.Module):
    def __init__(self,out_dim=2):
        super(Dattention_gated, self).__init__()
        self.embedding = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)
        self.head = nn.Linear(512,out_dim)
    def forward(self,x):
        x=x.squeeze()
        x = self.embedding(x) # 1024->512
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        x = torch.mm(A, x)  # KxL
        x = self.head(x.squeeze(1))

        return x



if __name__ == "__main__":
    model = Dattention_ori()
    # x=torch.rand(1,2,1024)
    # print(model(x))
    for k,v in model.state_dict().items():
        print(k)