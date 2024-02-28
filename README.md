# R$`^2`$T-MIL (Updating)
Official repo of **Feature Re-Embedding: Towards Foundation Model-Level Performance in Computational Pathology**, CVPR 2024. [[arXiv]](https://arxiv.org/abs/2402.17228)

## Training
```shell
python3 main.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --cv_fold=3 --model=rttmil --title=rttmil --n_trans_layers=1 --pool=attn --pos=none --da_act=tanh --attn=rrt --epeg_k=15 --seed=2021
```
