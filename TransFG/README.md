# TransFG: A Transformer Architecture for Fine-grained Recognition

## Best Results 
* 4/10: acc0.938356

## Experiments Results

1. (Acc: 0.92    ) Vit-B_16, smooth=0
2. (Acc: 0.926941) Vit-B_16, smooth=0.15 
2. (Acc: 0.936073) Vit-B_16, smooth=0.125 
2. (Acc: 0.92    ) Vit-B_16, smooth=0.1
3. **(Acc: 0.938356) Vit-B_16, smooth=0.07**
4. (Acc: 0.926941) Vit-B_16, smooth=0.05
5. (Acc: 0.926   ) Vit-B_16, smooth=0.07, CenterCrop (augmentation)
6. (Acc: 0.9155  ) Vit-B_32, smooth=0.07
7. (Acc: 0.75    ) ViT-L_16, smooth=0.07
> CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py  --split overlap --num_steps 10000 --fp16 --name ViT-L_16_smooth0.07_bsize1 --smoothing_value 0.07 --model_type ViT-L_16 --pretrained_dir pretrained_models/ViT-L_16.npz --eval_batch_size 4 --train_batch_size 1 --output_dir output_L16
8. (Acc: ) ViT-H_14, smooth=0.07
> CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py  --split overlap --num_steps 10000 --fp16 --name ViT-H_14_smooth0.07_bsize1 --smoothing_value 0.07 --model_type ViT-H_14 --pretrained_dir pretrained_models/ViT-H_14.npz --eval_batch_size 2 --train_batch_size 1 --output_dir output_output_H14
9. (Acc: 0.936073) Vit-B_16 (V2), smooth=0.07

> Remove "set_seed(args)"

10. (Acc: 0.926941) Vit-B_32 (V2), smooth=0.07
> CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py  --split overlap --num_steps 10000 --fp16 --name sample_smooth0.07_aug --smoothing_value 0.07 --model_type ViT-B_32 --pretrained_dir pretrained_models_V2/ViT-B_32.npz --output_dir output_B32_V2/
11. (Acc: 0.938356) Vit-B_16 (V2), smooth=0.07
12. (Acc: 0.922374) Vit-B_16 (V2), smooth=0.07, img_size 400 -> 224
13. (Acc: 0.920091) Vit-B_16 (V2), smooth=0.1, img_size 400 -> 224
14. (Acc: 0.915525) Vit-B_16 (V2), smooth=0.15, img_size 400 -> 224
15. (Acc: 0.920091) Vit-B_16 (V2), smooth=0.2, img_size 400 -> 224
16. (Acc: 0.938356) Vit-B_16, smooth=0.07, img_size 500 -> 350 
17. (Acc: 0.929224) Vit-B_16, smooth=0.1, img_size 500 -> 350 
> CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py  --split overlap --num_steps 10000 --fp16 --name sample_smooth0.1_img350 --smoothing_value 0.1 --model_type ViT-B_16 --pretrained_dir pretrained_models/ViT-B_16.npz --output_dir output --img_size 350 --train_batch_size 8

16. (Acc: 0.48858 ) cct_14_7x2_384
16. (Acc: 0.7169  ) cct_14_7x2_384_sine
16. (Acc:   ) cct_14_7x2_384_sine, img_size 500 -> 384 


## Dependencies:
+ Python 3.7.3
+ PyTorch 1.5.1
+ torchvision 0.6.1
+ ml_collections

## Usage
### 1. Download Google pre-trained ViT models

* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): ViT-B_16, ViT-B_32...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

```
1. unzip "training.zip" in "dataset" folder
2. python split_csv_file.py
```
```
-/dataset
    -| 1.jpg
    -| 2.jpg
    ...
    -|train_label.csv
    -|val_label.csv
    -|label.csv
    
```

### 3. Install required packages

Install dependencies with the following command:

```bash
pip3 install -r requirements.txt
```

### 4. Train

To train TransFG on CUB-200-2011 dataset with 4 gpus in FP-16 mode for 10000 steps run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py  --split overlap --num_steps 10000 --fp16 --name sample_smooth0.07 --smoothing_value 0.07 --model_type ViT-B_16 --pretrained_dir pretrained_models/ViT-B_16.npz --output_dir output
```

## Reference
[*TransFG: A Transformer Architecture for Fine-grained Recognition (AAAI2022)*](https://arxiv.org/abs/2103.07976)  


