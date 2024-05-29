import os
import time

from datasets.TCGA_Survival import TCGA_Survival

from utils.options import parse_args
from utils.util import set_seed
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.util import CV_Meter

from torch.utils.data import DataLoader, SubsetRandomSampler


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    # create results directory
    if args.evaluate:
        results_dir = args.resume
    elif args.model == "RRTMIL":
        results_dir = "./results/{dataset}/[{model}]-[{epeg_k}]-[{crmsa_k}]-[{folder}]-[{time}]".format(
            dataset=args.excel_file.split("/")[-1].split(".")[0],
            model=args.model,
            epeg_k=args.epeg_k,
            crmsa_k=args.crmsa_k,
            folder=args.folder,
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        )
    elif "IBMIL" in args.model:
        results_dir = "./results/{dataset}/[{model}]-[{k}]-[{folder}]-[{time}]".format(
            dataset=args.excel_file.split("/")[-1].split(".")[0],
            model=args.model,
            k=args.epeg_k,
            folder=args.folder,
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        )
    else:
        results_dir = "./results/{dataset}/[{model}]-[{folder}]-[{time}]".format(
            dataset=args.excel_file.split("/")[-1].split(".")[0],
            model=args.model,
            folder=args.folder,
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        )
    print(results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # define dataset
    args.num_classes = 4
    if args.folder == "plip":
        args.n_features = 512
    elif args.folder == "resnet50":
        args.n_features = 1024
    else:
        raise NotImplementedError("folder [{}] is not implemented".format(args.folder))
    dataset = TCGA_Survival(excel_file=args.excel_file, folder=args.folder)
    # 5-fold cross validation
    meter = CV_Meter(fold=5)
    # start 5-fold CV evaluation.
    for fold in range(5):
        # get split
        train_split, val_split = dataset.get_split(fold)
        train_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(train_split))
        val_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(val_split))

        # build model, criterion, optimizer, schedular
        #################################################
        # Unimodal: WSI
        if args.model == "IBMIL":
            # from models.IBMIL.network import TransMIL
            from models.IBMIL.network import Attention
            from models.IBMIL.engine import Engine

            confounder_path = "models/IBMIL/datasets_deconf/{}/{}/fold_{}".format(args.excel_file.split("/")[-1].split(".")[0].split("_")[0], args.folder, fold)
            # model = TransMIL(n_classes=args.n_features, input_size=args.n_features, confounder_path=confounder_path, k=args.epeg_k)
            model = Attention(in_size=args.n_features, out_size=args.num_classes, confounder_path=confounder_path, confounder_learn=True, k=args.epeg_k)
            engine = Engine(args, results_dir, fold)
        elif args.model == "DSMIL":
            from models.DSMIL.network import MILNet
            from models.DSMIL.engine import Engine

            model = MILNet(n_classes=args.num_classes, input_dim=args.n_features)
            engine = Engine(args, results_dir, fold)
        elif args.model == "MeanMIL":
            from models.Mean_Max.network import MeanMIL
            from models.Mean_Max.engine import Engine

            model = MeanMIL(n_classes=args.num_classes, dropout=True, act="relu", n_features=args.n_features)
            engine = Engine(args, results_dir, fold)
        elif args.model == "MaxMIL":
            from models.Mean_Max.network import MaxMIL
            from models.Mean_Max.engine import Engine

            model = MaxMIL(n_classes=args.num_classes, dropout=True, act="relu", n_features=args.n_features)
            engine = Engine(args, results_dir, fold)
        elif args.model == "AttMIL":
            from models.AttMIL.network import DAttention
            from models.AttMIL.engine import Engine

            model = DAttention(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
            engine = Engine(args, results_dir, fold)
        elif args.model == "CLAM_SB":
            from models.CLAM.network import CLAM_SB
            from models.CLAM.engine import Engine

            model = CLAM_SB(n_classes=args.num_classes, n_features=args.n_features)
            engine = Engine(args, results_dir, fold)
        elif args.model == "CLAM_MB":
            from models.CLAM.network import CLAM_MB
            from models.CLAM.engine import Engine

            model = CLAM_MB(n_classes=args.num_classes, n_features=args.n_features)
            engine = Engine(args, results_dir, fold)
        elif args.model == "TransMIL":
            from models.TransMIL.network import TransMIL
            from models.TransMIL.engine import Engine

            model = TransMIL(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
            engine = Engine(args, results_dir, fold)
        elif args.model == "RRTMIL":
            from models.RRTMIL.network import RRT
            from models.RRTMIL.engine import Engine

            model = RRT(n_classes=args.num_classes, epeg_k=args.epeg_k, crmsa_k=args.crmsa_k,input_dim=args.n_features, region_num=16)
            engine = Engine(args, results_dir, fold)
        elif args.model == "DTFD":
            from models.DTFD.network import DTFD
            from models.DTFD.engine import Engine

            model = DTFD(1e-5, 1e-5, 100, input_dim=args.n_features, n_classes=args.num_classes, criterion=define_loss(args))
            engine = Engine(args, results_dir, fold)
        elif args.model == "MHIM-MIL":
            from models.MHIM.network import MHIM_MIL
            from models.MHIM.engine import Engine

            teacher_init_path = None
            model_param = {
                "num_epoch": args.num_epoch,
                "niter_per_ep": len(train_loader),
                "input_dim": args.n_features,
                "mlp_dim": 512,
                "n_classes": args.num_classes,
                "mask_ratio": 0.7,
                "mask_ratio_l": 0.2,
                "mask_ratio_h": 0.02,
                "baseline": "attn",
                "teacher_init": teacher_init_path,
            }
            model = MHIM_MIL(**model_param)
            engine = Engine(args, results_dir, fold)
        else:
            raise NotImplementedError("model [{}] is not implemented".format(args.model))
        print("[model] trained model: ", args.model)
        criterion = define_loss(args)
        print("[model] loss function: ", args.loss)
        optimizer = define_optimizer(args, model)
        print("[model] optimizer: ", args.optimizer)
        scheduler = define_scheduler(args, optimizer)
        print("[model] scheduler: ", args.scheduler)
        # start training
        score, epoch = engine.learning(model, train_loader, val_loader, criterion, optimizer, scheduler)
        meter.updata(score, epoch)

    csv_path = os.path.join(results_dir, "results_{}.csv".format(args.model))
    meter.save(csv_path)


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
