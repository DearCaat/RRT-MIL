import os
import csv
import torch
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
# import sys
# from models.utils import *
# sys.stdout = open('data1.log', mode='w', encoding='utf-8')
def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def get_patient_label(csv_file):
    patients_list=[]
    labels_list=[]
    label_file = readCSV(csv_file)
    for i in range(0, len(label_file)):
        patients_list.append(label_file[i][0])
        labels_list.append(label_file[i][1])
    a=Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("all_counter:{}".format(dict(a)))
    return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object)

def data_split(full_list, ratio, shuffle=True,label=None,label_balance_val=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    # 根据类别比例选定验证集
    if label_balance_val and label is not None:
        _label = label[full_list]
        _label_uni = np.unique(_label)
        sublist_1 = []
        sublist_2 = []

        for _l in _label_uni:
            _list = full_list[_label == _l]
            n_total = len(_list)
            offset = int(n_total * ratio)
            if shuffle:
                random.shuffle(_list)
            sublist_1.extend(_list[:offset])
            sublist_2.extend(_list[offset:])

    else:
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        if shuffle:
            random.shuffle(full_list)
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]

    return sublist_1, sublist_2


def get_kflod(k, patients_array, labels_array,val_ratio=False,label_balance_val=True):
    if k > 1:
        skf = StratifiedKFold(n_splits=k)
    else:
        raise NotImplementedError
    train_patients_list = []
    train_labels_list = []
    test_patients_list = []
    test_labels_list = []
    val_patients_list = []
    val_labels_list = []
    for train_index, test_index in skf.split(patients_array, labels_array):
        if val_ratio != 0.:
            val_index,train_index = data_split(train_index,val_ratio,True,labels_array,label_balance_val)
            x_val, y_val = patients_array[val_index], labels_array[val_index]
        else:
            x_val, y_val = [],[]
        x_train, x_test = patients_array[train_index], patients_array[test_index]
        y_train, y_test = labels_array[train_index], labels_array[test_index]

        train_patients_list.append(x_train)
        train_labels_list.append(y_train)
        test_patients_list.append(x_test)
        test_labels_list.append(y_test)
        val_patients_list.append(x_val)
        val_labels_list.append(y_val)
        
    # print("get_kflod.type:{}".format(type(np.array(train_patients_list))))
    return np.array(train_patients_list,dtype=object), np.array(train_labels_list,dtype=object), np.array(test_patients_list,dtype=object), np.array(test_labels_list,dtype=object),np.array(val_patients_list,dtype=object), np.array(val_labels_list,dtype=object)

def get_tcga_parser(root,cls_name,mini=False):
        x = []
        y = []

        for idx,_cls in enumerate(cls_name):
            _dir = 'mini_pt' if mini else 'pt_files'
            _files = os.listdir(os.path.join(root,_cls,'features',_dir))
            _files = [os.path.join(os.path.join(root,_cls,'features',_dir,_files[i])) for i in range(len(_files))]
            x.extend(_files)
            y.extend([idx for i in range(len(_files))])
            
        return np.array(x).flatten(),np.array(y).flatten()

class TCGADataset(Dataset):
    
    def __init__(self, file_name=None, file_label=None,max_patch=-1,root=None,persistence=True):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(TCGADataset, self).__init__()

        self.patient_name = file_name
        self.patient_label = file_label
        self.max_patch = max_patch
        self.root = root
        self.all_pts = os.listdir(os.path.join(self.root,'pt_files'))
        self.slide_name = []
        self.slide_label = []
        self.persistence = persistence

        for i,_patient_name in enumerate(self.patient_name):
            _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
            _ids = np.where(_sides != '0')[0]
            for _idx in _ids:
                if persistence:
                    self.slide_name.append(torch.load(os.path.join(self.root,'pt_files',_sides[_idx])))
                else:
                    self.slide_name.append(_sides[_idx])
                self.slide_label.append(self.patient_label[i])
        self.slide_label = [ 0 if _l == 'LUAD' else 1 for _l in self.slide_label]

    def __len__(self):
        return len(self.slide_name)

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        file_path = self.slide_name[idx]
        # file_path = self.csv_file[idx] # ['1_1.png']
        # patient_path = file_path[1]
        label = self.slide_label[idx]

        if self.persistence:
            features = file_path
        else:
            features = torch.load(os.path.join(self.root,'pt_files',file_path))
        #features = os.path.join(self.root,'pt_files',file_path)
        return features , int(label)

class C16Dataset(Dataset):

    def __init__(self, file_name, file_label,root,persistence=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(C16Dataset, self).__init__()
        # self.csv_file = readCSV(csv_file)
        self.file_name = file_name
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = os.path.join(self.root,"pt")
            file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
            # file_path = self.csv_file[idx] # ['1_1.png']
            # patient_path = file_path[1]
            features = torch.load(file_path)
        label = int(self.slide_label[idx])
        # num = features.shape[0]
        # if num > 128:
        #     sample = np.random.choice(num, 128, replace=False)
        # else:
        #     sample = np.random.choice(num, num, replace=False)
        # sample = np.sort(sample)
        # new_feat = features[sample]
        # return new_feat, label

        return features , label
if __name__ == "__main__":
    # random.seed(2021)
    # label_path='/home/xxx/data/TransMIL/label.csv'
    label_path = '/data/xxx/c16_clam_bio_seg/label.csv'
    # --->划分数据集
    p, l = get_patient_label(label_path)
    index = [i for i in range(len(p))]
    random.shuffle(index)
    p = p[index]
    l = l[index]
    # train_p, train_l, test_p, test_l = get_kflod(3, p, l)
    train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(3, p, l,0)
    generator = torch.Generator()
    train_set = C16Dataset(val_p[0],val_l[0],root="/data/xxx/c16_clam_bio_seg/")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2,generator=generator)
    for i, data in enumerate(train_loader):

        bag=data[0]  # b*n*1024
        label=data[1]
        print(bag.shape,label)  # label->tensor([0])
    
    # print(test_p[0])
    # for k in range(3):
    #     print("\r开始第{}折".format(k))
    #     test_label=[]
    #     for i in range(len(test_p[k])):
    #         if test_p[k][i].split('_')[0]=='test':
    #             print(test_p[k][i],test_l[k][i])
    #             test_label.append(test_l[k][i])
    #     print(Counter(test_label))

    # # 创建训练测试文件
    # data={'path':[], 'label':[]}
    # for i in range(len(p)):
    #     image_path = "/data/yangpingan/proxy/camelyon16/TransMIL/pt/" + p[i] + ".pt"
    #     data['path'].append(image_path)
    #     data['label'].append(l[i])
    # data_frame = pd.DataFrame(data=data)
    # data_frame.to_csv("/data/yangpingan/proxy/camelyon16/TransMIL/sample_label/test.csv")
    # test_set = MyDataset(test_p[2],test_l[2])  
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4,
    #                         pin_memory=False)
    # model = contrastive1.MAE().cuda()
    # for i, data in enumerate(test_loader):
    #     bag=data[0].cuda()
    #     out = model(bag)
    #     # aug_bag=data[1].squeeze()
    #     # label=data[2].type(torch.float)
    #     print(out)
