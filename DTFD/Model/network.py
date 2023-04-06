import os
import torch
import torch.nn as nn
import sys
sys.path.append('..')
# sys.path.append('/home/xxx/code/mil/cvpr2023/modules/rrt')
#from modules.rrt import RRTEncoder

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0,n_robust=0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)
        
        self.apply(initialize_weights)

        if n_robust>0:
            # 改变init
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)
            # 改变后续的随机状态
            [torch.rand((1024,512)) for i in range(n_robust)]

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512,n_robust=0):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
      
        self.apply(initialize_weights)

        if n_robust>0:
            # 改变init
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)
            # 改变后续的随机状态
            [torch.rand((1024,512)) for i in range(n_robust)]
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0,dropout=False,act='relu',n_robust=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        # 添加自己的re-embedding模块
        #self.rrt = RRTEncoder(attn='rrt',pool='none',region_size=0)
        self.relu1 = nn.ReLU(inplace=True) if act.lower() == 'relu' else nn.GELU()
        self.numRes = numLayer_Res
        self.drop = nn.Dropout(0.25)
        self.dropout = dropout
        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

        
        self.apply(initialize_weights)

        if n_robust>0:
            # 改变init
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)
            # 改变后续的随机状态
            [torch.rand((1024,512)) for i in range(n_robust)]
    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)
        if self.dropout:
            x = self.drop(x)
        if self.numRes > 0:
            x = self.resBlocks(x)
        # x = self.rrt(x)

        return x
if __name__ == "__main__":
    initfc = "../debug_log/bio_dtfd_ourpra/best_model.pth"
    pre_dic = torch.load(initfc)
    
    model = DimReduction(n_channels=1024)
    model.load_state_dict(pre_dic['dim_reduction'])
    for k,v in model.state_dict().items():
        print(v)





