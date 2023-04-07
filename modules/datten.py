import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,input_dim=512,act='relu',bias=False,dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D,bias=bias)]

        if act == 'gelu': 
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K,bias=bias)]

        self.attention = nn.Sequential(*self.attention)

    def forward(self,x,no_norm=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)
        
        if no_norm:
            return x,A_ori
        else:
            return x,A

class AttentionGated(nn.Module):
    def __init__(self,input_dim=512,act='relu',bias=False,dropout=False):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

    def forward(self, x,no_norm=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        if no_norm:
            return x,A_ori
        else:
            return x,A

class DAttention(nn.Module):
    def __init__(self,input_dim=512,act='relu',gated=False,bias=False,dropout=False):
        super(DAttention, self).__init__()
        self.gated = gated
        if gated:
            self.attention = AttentionGated(input_dim,act,bias,dropout)
        else:
            self.attention = Attention(input_dim,act,bias,dropout)

    def forward(self, x, return_attn=False,no_norm=False,**kwargs):

        x,attn = self.attention(x,no_norm)

        if return_attn:
            return x.squeeze(1),attn.squeeze(1)
        else:   
            return x.squeeze(1)