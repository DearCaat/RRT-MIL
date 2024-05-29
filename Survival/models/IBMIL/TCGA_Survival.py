import os
import random
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data


class TCGA_Survival(data.Dataset):
    def __init__(self, excel_file, modal, signatures=None, folder='plip'):
        self.modal = modal
        self.folder = folder
        self.signatures = signatures
        # for grouping gene sequence
        if self.signatures:
            csv = pd.read_csv(self.signatures)
            self.omic_names = []
            for col in csv.columns:
                omic = csv[col].dropna().unique()
                self.omic_names.append(omic)
        print('[dataset] loading dataset from %s' % (excel_file))
        rows = pd.read_csv(excel_file)
        self.rows = self.disc_label(rows)
        self.omics_size = self.len_omics(self.rows)
        print('[dataset] sizes of omics data: ', self.omics_size)
        label_dist = self.rows['Label'].value_counts().sort_index()
        print('[dataset] discrete label distribution: ')
        print(label_dist)
        print('[dataset] required modality: %s' % (modal))
        for key in modal.split('_'):
            if key not in ['WSI', 'Gene']:
                raise NotImplementedError('modality [{}] is not implemented'.format(modal))
        print('[dataset] dataset from %s, number of cases=%d' % (excel_file, len(self.rows)))

    # def get_split(self, fold=0, keep_missing=True, require_label=True):
    #     assert 0 <= fold <= 4, 'fold should be in 0 ~ 4'
    #     split = self.rows['Fold {}'.format(fold)].values.tolist()
    #     missing = [i for i, x in enumerate(split) if x == 'train: missing']
    #     train_split = [i for i, x in enumerate(split) if x == 'train: complete']
    #     val_split = [i for i, x in enumerate(split) if x == 'val: complete']
    #     if keep_missing:
    #         if self.modal == 'WSI':
    #             for idx in range(len(missing)):
    #                 if self.rows.loc[missing[idx], 'Data_WSI'] == 1 and self.rows.loc[missing[idx], 'Data_Event'] == 1:
    #                     train_split.append(missing[idx])
    #         elif self.modal == 'Gene':
    #             for idx in range(len(missing)):
    #                 if self.rows.loc[missing[idx], 'Data_Gene'] == 1 and self.rows.loc[missing[idx], 'Data_Event'] == 1:
    #                     train_split.append(missing[idx])
    #         elif self.modal == 'WSI_Gene':
    #             if require_label:
    #                 for idx in range(len(missing)):
    #                     if self.rows.loc[missing[idx], 'Data_Event'] == 1:
    #                         train_split.append(missing[idx])
    #             else:
    #                 train_split += missing
    #         else:
    #             raise NotImplementedError('modality [{}] is not implemented'.format(self.modal))
    #         print("[dataset] (fold {}) training split (include missing data): {}, validation split: {}".format(fold, len(train_split), len(val_split)))
    #     else:
    #         print("[dataset] (fold {}) training split (exclude missing data): {}, validation split: {}".format(fold, len(train_split), len(val_split)))
    #     return train_split, val_split

    def get_split(self, fold=0):
        random.seed(1)
        assert 0 <= fold <= 4, 'fold should be in 0 ~ 4'
        split = self.rows['Fold {}'.format(0)].values.tolist()
        all_split = [i for i, x in enumerate(split) if x == 'train: complete']
        all_split += [i for i, x in enumerate(split) if x == 'val: complete']
        missing = [i for i, x in enumerate(split) if x == 'train: missing']
        for idx in range(len(missing)):
            if self.rows.loc[missing[idx], 'Data_WSI'] == 1 and self.rows.loc[missing[idx], 'Data_Event'] == 1:
                all_split.append(missing[idx])
        all_split = random.sample(all_split, len(all_split))
        if fold < 4:
            val_split = all_split[fold * (round(len(all_split) / 5)): (fold + 1) * (round(len(all_split) / 5))]
        else:
            val_split = all_split[fold * (round(len(all_split) / 5)):]
        train_split = [i for i in all_split if i not in val_split]
        return train_split, val_split

    def read_WSI(self, path):
        if str(path) == 'nan':
            return torch.zeros((1))
        else:
            path = path.replace('resnet50', self.folder)
            wsi = [torch.load(x) for x in path.split(';')]
            wsi = torch.cat(wsi, dim=0)
            # print('######################', wsi.shape)
            return wsi

    def read_Omics(self, path):
        if str(path) == 'nan':
            return torch.zeros((1))
        else:
            genes = pd.read_csv(path)
            if self.signatures:
                omic1 = torch.from_numpy(np.array(genes[genes['Gene'].isin(self.omic_names[0])]['Value'].values.tolist()).astype(np.float32))
                omic2 = torch.from_numpy(np.array(genes[genes['Gene'].isin(self.omic_names[1])]['Value'].values.tolist()).astype(np.float32))
                omic3 = torch.from_numpy(np.array(genes[genes['Gene'].isin(self.omic_names[2])]['Value'].values.tolist()).astype(np.float32))
                omic4 = torch.from_numpy(np.array(genes[genes['Gene'].isin(self.omic_names[3])]['Value'].values.tolist()).astype(np.float32))
                omic5 = torch.from_numpy(np.array(genes[genes['Gene'].isin(self.omic_names[4])]['Value'].values.tolist()).astype(np.float32))
                omic6 = torch.from_numpy(np.array(genes[genes['Gene'].isin(self.omic_names[5])]['Value'].values.tolist()).astype(np.float32))
                return (omic1, omic2, omic3, omic4, omic5, omic6)
            else:
                key = genes['Gene'].values.tolist()
                index = torch.from_numpy(np.array(genes['Index'].values.tolist()).astype(np.float32))
                value = torch.from_numpy(np.array(genes['Value'].values.tolist()).astype(np.float32))
                return (key, index, value)

    def __getitem__(self, index):
        case = self.rows.iloc[index, :].values.tolist()
        Study, ID, Event, Status, WSI, RNA = case[:6]
        Label = case[-1]
        Censorship = 1 if int(Status) == 0 else 0
        if self.modal == 'WSI':
            WSI = self.read_WSI(WSI)
            return (ID, WSI, Event, Censorship, Label)
        elif self.modal == 'Gene':
            RNA = self.read_Omics(RNA)
            return (ID, RNA, Event, Censorship, Label)
        elif self.modal == 'WSI_Gene':
            WSI = self.read_WSI(WSI)
            RNA = self.read_Omics(RNA)
            return (ID, WSI, RNA, Event, Censorship, Label)
        else:
            raise NotImplementedError('modality [{}] is not implemented'.format(self.modal))

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

    def len_omics(self, rows):
        for omics in rows['RNA'].values.tolist():
            if os.path.exists(str(omics)):
                data = pd.read_csv(omics)
                if self.signatures:
                    omic1 = data[data['Gene'].isin(self.omic_names[0])]['Value'].values.tolist()
                    omic2 = data[data['Gene'].isin(self.omic_names[1])]['Value'].values.tolist()
                    omic3 = data[data['Gene'].isin(self.omic_names[2])]['Value'].values.tolist()
                    omic4 = data[data['Gene'].isin(self.omic_names[3])]['Value'].values.tolist()
                    omic5 = data[data['Gene'].isin(self.omic_names[4])]['Value'].values.tolist()
                    omic6 = data[data['Gene'].isin(self.omic_names[5])]['Value'].values.tolist()
                    return (len(omic1), len(omic2), len(omic3), len(omic4), len(omic5), len(omic6))
                else:
                    return len(data.iloc[:, 0].values.tolist())


if __name__ == '__main__':
    from torch.utils.data import DataLoader, SubsetRandomSampler
    dataset = TCGA_Survival('/master/zhou_feng_tao/code/MissSurv/csv/Cbioportal/{}_Splits.csv'.format('BRCA'), modal='WSI', signatures='/master/zhou_feng_tao/code/MissSurv/csv/signatures.csv')
    train_split, val_split = dataset.get_split(fold=0)
    print(len(train_split), len(val_split))
    print(train_split)
    print(val_split)
    train_split, val_split = dataset.get_split(fold=1)
    print(len(train_split), len(val_split))
    print(train_split)
    print(val_split)
    # for study in ['BLCA', 'BRCA', 'CESC', 'CRC', 'ESCA', 'GBMLGG', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PAAD', 'PCPG', 'PRAD', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC']:
    #     dataset = TCGA_Survival('/master/zhou_feng_tao/code/MissSurv/csv/Cbioportal/{}_Splits.csv'.format(study), modal='Gene', signatures='/master/zhou_feng_tao/code/MissSurv/csv/signatures.csv')
    #     dataset.get_split(fold=0, keep_missing=True)
    # print(dataset.omics_size)
    # dataset = TCGA_Survival('/master/zhou_feng_tao/code/MissSurv/csv/Cbioportal/BLCA_Splits.csv', modal='Gene', signatures='/master/zhou_feng_tao/code/MissSurv/csv/signatures.csv')
    # print(dataset.omics_size)
    # train_split, val_split = dataset.get_split(fold=0, keep_missing=True)
    # train_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(train_split))
    # val_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(val_split))
    # for idx, (ID, RNA, Event, Status, Label) in enumerate(train_loader):
    #     print(ID)
    #     print(RNA[0].shape, RNA[1].shape, RNA[2].shape, RNA[3].shape, RNA[4].shape, RNA[5].shape)
