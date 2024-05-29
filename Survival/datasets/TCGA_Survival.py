import os
import random
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data


class TCGA_Survival(data.Dataset):
    def __init__(self, excel_file, folder='plip'):
        self.folder = folder
        print('[dataset] loading dataset from %s' % (excel_file))
        rows = pd.read_csv(excel_file)
        self.rows = self.disc_label(rows)
        label_dist = self.rows['Label'].value_counts().sort_index()
        print('[dataset] discrete label distribution: ')
        print(label_dist)
        print('[dataset] dataset from %s, number of cases=%d' % (excel_file, len(self.rows)))

    def get_split(self, fold=0):
        random.seed(1)
        ratio=0.2
        assert 0 <= fold <= 4, 'fold should be in 0 ~ 4'
        sample_index = random.sample(range(len(self.rows)), len(self.rows))
        num_split = round((len(self.rows) - 1) * ratio)
        if fold < 1 / ratio - 1:
            val_split = sample_index[fold * num_split: (fold + 1) * num_split]
        else:
            val_split = sample_index[fold * num_split:]
        train_split = [i for i in sample_index if i not in val_split]
        print("[dataset] training split: {}, validation split: {}".format(len(train_split), len(val_split)))
        return train_split, val_split 
    
    def read_WSI(self, path):
        path = path.replace('resnet50', self.folder)
        wsi = [torch.load(x) for x in path.split(';')]
        wsi = torch.cat(wsi, dim=0)
        return wsi

    def __getitem__(self, index):
        case = self.rows.iloc[index, :].values.tolist()
        Study, ID, Event, Status, WSI = case[:5]
        Label = case[-1]
        Censorship = 1 if int(Status) == 0 else 0
        WSI = self.read_WSI(WSI)
        return (ID, WSI, Event, Censorship, Label)

    def __len__(self):
        return len(self.rows)

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows['Status'] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows['Event'].max() + eps
        q_bins[0] = rows['Event'].min() - eps
        disc_labels, q_bins = pd.cut(rows['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        # missing event data
        disc_labels = disc_labels.values.astype(int)
        disc_labels[disc_labels < 0] = -1
        rows.insert(len(rows.columns), 'Label', disc_labels)
        return rows


if __name__ == '__main__':
    dataset = TCGA_Survival('/master/zhou_feng_tao/code/Survival/csv/{}_Splits.csv'.format('LUSC'))
    train_split, val_split = dataset.get_split(fold=0)
    print(len(train_split), len(val_split))
    print(train_split)
    print(val_split)
    train_split, val_split = dataset.get_split(fold=1)
    print(len(train_split), len(val_split))
    print(train_split)
    print(val_split)