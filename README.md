# R$`^2`$T-MIL (Updating)
Official repo of **Feature Re-Embedding: Towards Foundation Model-Level Performance in Computational Pathology**, CVPR 2024. [[arXiv]](https://arxiv.org/abs/2402.17228)

## TODO
- Uploading all codes
- Uploading all datasets

## Prepare Patch Features
To preprocess WSIs, we used [CLAM](https://github.com/mahmoodlab/CLAM/tree/master#wsi-segmentation-and-patching).

Download the preprocessed patch features (Updating): [Baidu Cloud](https://pan.baidu.com/s/10NKByw7Txk4Vqc3UoN0qvQ?pwd=2023).

## Training
Download the Docker Image: [Baidu Cloud](https://pan.baidu.com/s/1EN1JUbIjAl73NwHZF3YlPA?pwd=fek8).

```shell
python3 main.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --cv_fold=3 --model=rttmil --title=rttmil --n_trans_layers=1 --pool=attn --pos=none --da_act=tanh --attn=rrt --epeg_k=15 --seed=2021
```
## Citing R$`^2`$T-MIL
```
@misc{tang2024feature,
      title={Feature Re-Embedding: Towards Foundation Model-Level Performance in Computational Pathology}, 
      author={Wenhao Tang and Fengtao Zhou and Sheng Huang and Xiang Zhu and Yi Zhang and Bo Liu},
      year={2024},
      eprint={2402.17228},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
