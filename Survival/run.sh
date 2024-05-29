studies="BLCA LUAD LUSC"
models="DTFD TransMIL AttMIL MaxMIL MHIM-MIL"
for study in $studies
do
    for model in $models
    do
        CUDA_VISIBLE_DEVICES=3 python main.py --model $model \
                                              --excel_file ./csv/${study}_Splits.csv \
                                              --num_epoch 30 \
                                              --batch_size 1 \
                                              --folder resnet50
    done
done