import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
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

class FCLayer(nn.Module):
    def __init__(self, dropout=0.25,act='relu',in_size=1024,rrt=None):
        super(FCLayer, self).__init__()
        self.embed = [nn.Linear(in_size, 512)]
        # self.embed.append(SwinEncoder(attn='swin',pool='none'))
        # self.embed = nn.ModuleList([nn.Linear(1024, 512)])
        
        if act.lower() == 'gelu':
            self.embed += [nn.GELU()]
        else:
            self.embed += [nn.ReLU()]

        if dropout:
            self.embed += [nn.Dropout(dropout)]
        self.rrt = rrt if rrt is not None else nn.Identity()
        self.embed = nn.Sequential(*self.embed)

    def forward(self, feats):
        feats = self.embed(feats)

        return self.rrt(feats)

class Dattention_ori(nn.Module):
    def __init__(self,out_dim=2,in_size=1024,dropout=0.25,confounder_path=False,**kwargs):
        super(Dattention_ori,self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.embedding = FCLayer(in_size=in_size,dropout=dropout,**kwargs)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.head = nn.Linear(512,out_dim)

        if confounder_path: 
            print('deconfounding')
            self.confounder_path = confounder_path
            conf_list = []
            if isinstance(confounder_path,list):
                for i in confounder_path:
                    conf_list.append(torch.from_numpy(np.load(i)).view(-1,512).float())
            else:
                conf_list.append(torch.from_numpy(np.load(confounder_path)).view(-1,512).float())
            conf_tensor = torch.cat(conf_list, 0) 
            conf_tensor_dim = conf_tensor.shape[-1]
            self.register_buffer("confounder_feat",conf_tensor)
            joint_space_dim = 128
            dropout_v = 0.5
            # self.confounder_W_q = nn.Linear(in_size, joint_space_dim)
            # self.confounder_W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            self.W_q = nn.Linear(512, joint_space_dim)
            self.W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            self.head =  nn.Linear(self.L*self.K+conf_tensor_dim, out_dim)
            self.dropout = nn.Dropout(dropout_v)

        self.apply(initialize_weights)
        
    def forward(self,x):
        x=x.squeeze()
        x = self.embedding(x) # 1024->512
        A = self.attention(x)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        x = torch.mm(A, x)  # KxL

        if self.confounder_path:
            device = x.device
            # bag_q = self.confounder_W_q(M)
            # conf_k = self.confounder_W_k(self.confounder_feat)
            bag_q = self.W_q(x)
            conf_k = self.W_k(self.confounder_feat)
            deconf_A = torch.mm(conf_k, bag_q.transpose(0, 1))
            deconf_A = F.softmax( deconf_A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
            conf_feats = torch.mm(deconf_A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
            x = torch.cat((x,conf_feats),dim=1)

        x = self.head(x)
        return x

if __name__ == "__main__":
    x=torch.rand(5,3,64,64).cuda()

