# RRT-MIL
The part of training codes of CVPR2024 submission 4372

## Training
```shell
python3 main.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --cv_fold=3 --model=rttmil --title=rttmil --n_trans_layers=1 --pool=attn --pos=none --da_act=tanh --attn=rrt --epeg_k=15 --seed=2021
```
