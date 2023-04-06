import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .swin import SwinEncoder
# import sys
# sys.stdout = open('data1.log', mode='w', encoding='utf-8')

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
# class FCLayer(nn.Module):
#     def __init__(self, in_size, out_size=1):
#         super(FCLayer, self).__init__()
#         self.embed = nn.Sequential(nn.Linear(1024, 512))
#         self.fc = nn.Sequential(nn.Linear(in_size, out_size))
#     def forward(self, feats):
#         feats = self.embed(feats)
#         x = self.fc(feats)
#         return feats, x

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1,dropout=True,act='relu'):
        super(FCLayer, self).__init__()

        self.embed = [nn.Linear(1024, 512)]
    
        if act.lower() == 'gelu':
            self.embed += [nn.GELU()]
        else:
            self.embed += [nn.ReLU()]

        if dropout:
            self.embed += [nn.Dropout(0.25)]
        self.embed.append(SwinEncoder(attn='swin',pool='none'))

        self.embed = nn.Sequential(*self.embed)
        self.fc = nn.Sequential(
            nn.Linear(in_size, out_size))
    def forward(self, feats):
        feats = self.embed(feats)
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        feats = feats.squeeze()
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=False, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier,n_robust=0):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

        self.apply(initialize_weights)
        if n_robust>0:
            # 改变init
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)
            # 改变后续的随机状态
            [torch.rand((1024,512)) for i in range(n_robust)]

        # if initfc:
        #     pre_dict = torch.load(initfc)
        #     new_state_dict ={}
        #     target = ['i_classifier.embed.0.weight','i_classifier.embed.0.bias']
        #     for k,v in pre_dict.items():
        #         if k in target:
        #             new_state_dict[k.split('.',1)[1]]=v
        #     self.i_classifier.load_state_dict(new_state_dict,strict=False)
        #     print('embedding fc Inited')
            # print(info)

    def forward(self, x):
        feats, classes = self.i_classifier(x.squeeze()) # feats->N x D
        # feats, classes = self.i_classifier(x) # feats->N x D 添加swin需要b*n*c的尺度
        prediction_bag, A, B = self.b_classifier(feats, classes)
        max_prediction, _ = torch.max(classes, 0) 
        
        # return classes, prediction_bag, A, B
        return max_prediction, prediction_bag

if __name__ == "__main__":
    feats_size = 512
    num_classes = 2
    dropout_node = 0
    non_linearity = 1
    def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    

    seed_torch(seed=2022)
#     criterion = nn.BCEWithLogitsLoss()
    i_classifier = FCLayer(in_size=feats_size, out_size=num_classes)
    b_classifier = BClassifier(input_size=feats_size, output_class=num_classes, dropout_v=dropout_node, nonlinear=non_linearity)
    milnet = MILNet(i_classifier, b_classifier)
#     # # milnet = FCLayer(in_size=512)
#     # x=torch.rand(1,3,1024)
#     # label = torch.rand(1)
#     # max_prediction, bag_prediction = milnet(x)
#     # print(max_prediction.unsqueeze(dim=0).shape, bag_prediction.shape)
    init = "/data/xxx/output/dsmil/dsmil_reproduce_zxx/bio_avg_ourpar_fc/fold_1_model_best_auc.pt"
    # pre_dict = torch.load(init)
    # new_state_dict ={}
    # target = ['i_classifier.embed.0.weight']
    # for k,v in pre_dict.items():
    #     if k in target:
    #         new_state_dict[k]=v
    #         print(v)
        
    # info = milnet.load_state_dict(new_state_dict,strict=False)
    # for k,v in milnet.state_dict().items():
    #     if k =='i_classifier.embed.0.weight':
    #         print(v)
    # print('Teacher Inited')
#     print(info)
    model1 =  MILNet(i_classifier, b_classifier,initfc=init,test=1)
    for k,v in model1.state_dict().items():
        if k == 'b_classifier.fcc.weight':
            print(v)



