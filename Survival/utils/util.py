import os
import csv
import random
import numpy as np

import torch
from torch.utils.data import Sampler


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_weights_for_balanced_classes_split(dataset):
    num_classes = 4
    N = float(len(dataset))
    cls_ids = [[] for i in range(num_classes)]
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        cls_ids[label].append(idx)
    weight_per_class = [N / len(cls_ids[c]) for c in range(num_classes)]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        weight[idx] = weight_per_class[label]
    return torch.DoubleTensor(weight)


def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CV_Meter():
    def __init__(self, fold=5):
        self.fold = fold
        self.header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
        self.epochs = ["epoch"]
        self.cindex = ["cindex"]

    def updata(self, score, epoch):
        self.epochs.append(epoch)
        self.cindex.append(round(score, 4))

    def save(self, path):
        self.cindex.append(round(np.mean(self.cindex[1:self.fold + 1]), 4))
        self.cindex.append(round(np.std(self.cindex[1:self.fold + 1]), 4))
        print("save evaluation resluts to", path)
        with open(path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(self.header)
            writer.writerow(self.epochs)
            writer.writerow(self.cindex)
