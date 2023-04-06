import copy
import torch
import numpy as np
from torch import nn
from einops import repeat
import torchvision.models as models
from modules.emb_position import *
from modules.mlp import *
from modules.datten import *
from modules.rmsa import *
import torch.nn.functional as F
from modules.translayer import *
from modules.datten import DAttention
import math

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

class RRTEncoder(nn.Module):
    '''
    learnable mask token
    '''
    def __init__(self,mlp_dim=512,pos_pos=0,pos='none',peg_k=7,attn='ntrans',region_num=8,drop_out=0.1,n_layers=1,n_heads=8,multi_scale=False,drop_path=0.1,pool='attn',da_act='tanh',reduce_ratio=0,ffn=False,ffn_act='gelu',mlp_ratio=4.,da_gated=False,da_bias=False,da_dropout=False,trans_dim=64,n_cycle=1,trans_conv=True,rpe=False,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,shift_size=False,peg_bias=True,peg_1d=False,**kwargs):
        super(RRTEncoder, self).__init__()
        
        # 不需要降维
        if reduce_ratio == 0:
            pass
        # 根据多尺度自动降维，维度前后不变
        elif reduce_ratio == -1:
            reduce_ratio = n_layers-1
        # 指定降维维度 (2的n次)
        else:
            pass

        self.final_dim = mlp_dim // (2**reduce_ratio) if reduce_ratio > 0 else mlp_dim
        if multi_scale:
            self.final_dim = self.final_dim * (2**(n_layers-1))

        self.pool = pool
        if pool == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
        elif pool == 'attn':
            self.pool_fn = DAttention(self.final_dim,da_act,gated=da_gated,bias=da_bias,dropout=da_dropout)

        self.norm = nn.LayerNorm(self.final_dim)

        # if attn == 'trans':
        self.layer1 = TransLayer1(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,need_down=multi_scale,need_reduce=reduce_ratio!=0,down_ratio=2**reduce_ratio,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,n_cycle=n_cycle,attn=attn,n_region=region_num,trans_conv=trans_conv,rpe=rpe,region_size=region_size,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias,shift_size=shift_size,**kwargs)
        #self.layer2 = TransLayer1(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path)
        if reduce_ratio > 0:
            mlp_dim = mlp_dim // (2**reduce_ratio)

        if multi_scale:
            mlp_dim = mlp_dim*2

        if n_layers >= 2:
            self.layers = []
            for i in range(n_layers-2):
                self.layers += [TransLayer1(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,need_down=multi_scale,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,n_cycle=n_cycle,attn=attn,n_region=region_num,trans_conv=trans_conv,rpe=rpe,region_size=region_size,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias) ]
                if multi_scale:
                    mlp_dim = mlp_dim*2
            self.layers += [TransLayer1(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,n_cycle=n_cycle,attn=attn,n_region=region_num,trans_conv=trans_conv,rpe=rpe,region_size=region_size,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias,shift_size=shift_size,**kwargs)]
            self.layers = nn.Sequential(*self.layers)
        else:
            self.layers = nn.Identity()

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

    def forward(self, x, no_pool=False,return_attn=False,no_norm=False):
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0),x.size(1),-1)
            x = x.transpose(1,2)
            shape_len = 4
        batch, num_patches, C = x.shape # 直接是特征
        patch_idx = 0
        if self.pos_pos == -2:
            x = self.pos_embedding(x)
        
        # cls_token
        if self.pool == 'cls_token':
            cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = batch)
            x = torch.cat((cls_tokens, x), dim=1)
            patch_idx = 1

        if self.pos_pos == -1:
            x[:,patch_idx:,:] = self.pos_embedding(x[:,patch_idx:,:])

        # translayer1
        x = self.layer1(x)
        #print(x)
        #加位置编码
        if self.pos_pos == 0:
            x[:,patch_idx:,:] = self.pos_embedding(x[:,patch_idx:,:])
        
        # translayer2
        for i,layer in enumerate(self.layers.children()):
            x = layer(x)

        #---->cls_token
        x = self.norm(x)

        if no_pool:
            if shape_len == 2:
                x = x.squeeze(0)
            elif shape_len == 4:
                x = x.transpose(1,2)
                x = x.reshape(batch,C,int(num_patches**0.5),int(num_patches**0.5))
            return x
            
        if self.pool == 'cls_token':
            logits = x[:,0,:]
        elif self.pool == 'avg':
            logits = x.mean(dim=1)
        elif self.pool == 'attn':
            if return_attn:
                logits,a = self.pool_fn(x,return_attn=True,no_norm=no_norm)
            else:
                logits = self.pool_fn(x)

        else:
            logits = x

        if shape_len == 2:
            logits = logits.squeeze(0)
        elif shape_len == 4:
            logits = logits.transpose(1,2)
            logits = logits.reshape(batch,C,int(num_patches**0.5),int(num_patches**0.5))

        if return_attn:
            return logits,a
        else:
            return logits

class RRT(nn.Module):
    def __init__(self, mlp_dim=512,act='relu',n_classes=2,dropout=0.25,pos_pos=0,n_robust=0,pos='ppeg',peg_k=7,attn='trans',pool='attn',region_num=8,n_layers=2,n_heads=8,multi_scale=False,drop_path=0.,da_act='relu',trans_dropout=0.1,ffn=False,ffn_act='gelu',mlp_ratio=4.,da_gated=False,da_bias=False,da_dropout=False,trans_dim=64,n_cycle=1,trans_conv=False,min_region_num=0,qkv_bias=True,shift_size=False,**kwargs):
        super(RRT, self).__init__()

        self.patch_to_emb = [nn.Linear(1024, 512)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        self.online_encoder = RRTEncoder(mlp_dim=mlp_dim,pos_pos=pos_pos,pos=pos,peg_k=peg_k,attn=attn,region_num=region_num,n_layers=n_layers,n_heads=n_heads,multi_scale=multi_scale,drop_path=drop_path,pool=pool,da_act=da_act,drop_out=trans_dropout,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,da_gated=da_gated,da_bias=da_bias,da_dropout=da_dropout,trans_dim=trans_dim,n_cycle=n_cycle,trans_conv=trans_conv,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias,shift_size=shift_size,**kwargs)

        self.predictor = nn.Linear(self.online_encoder.final_dim,n_classes)

        self.apply(initialize_weights)


        if n_robust>0:
            # 改变init
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)

            torch.random.set_rng_state(rng)
            # 改变后续的随机状态
            [torch.rand((1024,512)) for i in range(n_robust)]

    def forward(self, x, return_attn=False,no_norm=False):
        x = self.patch_to_emb(x) # n*512
        x = self.dp(x)
        
        ps = x.size(1)

        # forward online network
        if return_attn:
            x,a = self.online_encoder(x,return_attn=True,no_norm=no_norm)
        else:
            x = self.online_encoder(x)
        
        # prediction
        logits = self.predictor(x)

        if return_attn:
            return logits,a
        else:
            return logits
