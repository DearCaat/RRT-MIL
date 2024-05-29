import torch
import numpy as np
import time
from torch.utils.data import DataLoader, SubsetRandomSampler

torch.multiprocessing.set_sharing_strategy("file_system")
import os
import os
import time
import numpy as np
import faiss
import torch
from abmil import DAttention
from TCGA_Survival import TCGA_Survival


def preprocess_features(npdata, pca):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    assert npdata.dtype == np.float32

    if np.any(np.isnan(npdata)):
        raise Exception("nan occurs")
    if pca != -1:
        print("\nPCA from dim {} to dim {}".format(ndim, pca))
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
    if np.any(np.isnan(npdata)):
        percent = np.isnan(npdata).sum().item() / float(np.size(npdata)) * 100
        if percent > 0.1:
            raise Exception("More than 0.1% nan occurs after pca, percent: {}%".format(percent))
        else:
            npdata[np.isnan(npdata)] = 0.0
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)

    npdata = npdata / (row_sums[:, np.newaxis] + 1e-10)

    return npdata


def run_kmeans(x, nmb_clusters, verbose=False, seed=None):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    if seed is not None:
        clus.seed = seed
    else:
        clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]


class Kmeans:
    def __init__(self, k, pca_dim=256):
        self.k = k
        self.pca_dim = pca_dim

    def cluster(self, feat, verbose=False, seed=None):
        """Performs k-means clustering.
        Args:
            x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(feat, self.pca_dim)

        # cluster the data
        I = run_kmeans(xb, self.k, verbose, seed)
        self.labels = np.array(I)
        if verbose:
            print("k-means time: {0:.0f} s".format(time.time() - end))


def reduce(feats, k, dataset, feature, fold):
    """
    feats:bag feature tensor,[N,D]
    k: number of clusters
    shift: number of cov interpolation
    """
    prototypes = []
    semantic_shifts = []
    feats = feats.cpu().numpy()

    kmeans = Kmeans(k=k, pca_dim=-1)
    kmeans.cluster(feats, seed=66)  # for reproducibility
    assignments = kmeans.labels.astype(np.int64)
    # compute the centroids for each cluster
    centroids = np.array([np.mean(feats[assignments == i], axis=0) for i in range(k)])

    # compute covariance matrix for each cluster
    covariance = np.array([np.cov(feats[assignments == i].T) for i in range(k)])

    os.makedirs(f"datasets_deconf/{feature}/{dataset}_{fold}", exist_ok=True)
    prototypes.append(centroids)
    prototypes = np.array(prototypes)
    prototypes = prototypes.reshape(-1, 512)
    print(prototypes.shape)
    print(f"datasets_deconf/{feature}/{dataset}_{fold}/train_bag_cls_agnostic_feats_proto_{k}.npy")
    np.save(f"datasets_deconf/{feature}/{dataset}_{fold}/train_bag_cls_agnostic_feats_proto_{k}.npy", prototypes)

    del feats


def main():
    # define dataloader
    # resnet
    # ckpt = "/master/zhou_feng_tao/code/CVPR2024/results/WSI/BLCA_Splits/[AttMIL]-[2023-11-03]-[07-29-54]"
    # ckpt = "/master/zhou_feng_tao/code/CVPR2024/results/WSI/LUAD_Splits/[AttMIL]-[2023-11-01]-[08-51-06]"
    # ckpt = "/master/zhou_feng_tao/code/CVPR2024/results/WSI/LUSC_Splits/[AttMIL]-[resnet50]-[2023-11-10]-[20-30-51]"
    # dataset = TCGA_Survival(excel_file="/master/zhou_feng_tao/code/MissSurv/csv/Cbioportal/BLCA_Splits.csv", modal="WSI", folder="resnet50")
    # dataset = TCGA_Survival(excel_file="/master/zhou_feng_tao/code/MissSurv/csv/Cbioportal/LUAD_Splits.csv", modal="WSI", folder="resnet50")
    # dataset = TCGA_Survival(excel_file="/master/zhou_feng_tao/code/MissSurv/csv/Cbioportal/LUSC_Splits.csv", modal="WSI", folder="resnet50")
    # plip
    # ckpt = "/master/zhou_feng_tao/code/CVPR2024/results/WSI/BLCA_Splits/[AttMIL]-[2023-11-03]-[06-32-20]"
    # ckpt = "/master/zhou_feng_tao/code/CVPR2024/results/WSI/LUAD_Splits/[AttMIL]-[2023-11-02]-[11-51-57]"
    ckpt = "/master/zhou_feng_tao/code/CVPR2024/results/WSI/LUSC_Splits/[AttMIL]-[plip]-[2023-11-10]-[20-03-58]"
    # dataset = TCGA_Survival(excel_file="/master/zhou_feng_tao/code/MissSurv/csv/Cbioportal/BLCA_Splits.csv", modal="WSI", folder="plip")
    # dataset = TCGA_Survival(excel_file="/master/zhou_feng_tao/code/MissSurv/csv/Cbioportal/LUAD_Splits.csv", modal="WSI", folder="plip")
    dataset = TCGA_Survival(excel_file="/master/zhou_feng_tao/code/MissSurv/csv/Cbioportal/LUSC_Splits.csv", modal="WSI", folder="plip")

    for fold in range(5):
        # define model
        model = DAttention(n_classes=4, dropout=0.25, act="relu", n_features=512)
        for root, dirs, files in os.walk(os.path.join(ckpt, "fold_{}".format(fold))):
            for file in files:
                if file.endswith(".pth.tar"):
                    ckpt_cur = os.path.join(root, file)
        print("load checkpoint from {}".format(ckpt_cur))
        checkpoint = torch.load(ckpt_cur)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.cuda()
        model.eval()
        feats_list = []
        train_split, val_split = dataset.get_split(fold)
        train_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(train_split))
        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(train_loader):
            with torch.no_grad():
                data_WSI = data_WSI.cuda()
                bag_prediction, bag_feats, attention = model(data_WSI)
                print(bag_feats.shape)
                feats_list.append(bag_feats.cpu())
        bag_tensor = torch.cat(feats_list, dim=0)

        bag_tensor_ag = bag_tensor.view(-1, 512)
        for i in [2, 4, 8, 16]:
            # reduce(bag_tensor_ag, i, "BLCA", "resnet50", fold)
            # reduce(bag_tensor_ag, i, "LUAD", "resnet50", fold)
            # reduce(bag_tensor_ag, i, "LUSC", "resnet50", fold)
            # reduce(bag_tensor_ag, i, "BLCA", "plip", fold)
            # reduce(bag_tensor_ag, i, "LUAD", "plip", fold)
            reduce(bag_tensor_ag, i, "LUSC", "plip", fold)


if __name__ == "__main__":
    main()
