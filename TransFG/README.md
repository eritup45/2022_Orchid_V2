# TransFG: A Transformer Architecture for Fine-grained Recognition

## Best Results 
* 4/10: acc0.938356

## Experiments Results

1. (Acc: 0.92    ) Vit-B_16, smooth=0
2. (Acc: 0.92    ) Vit-B_16, smooth=0.1
3. **(Acc: 0.938356) Vit-B_16, smooth=0.07**
4. (Acc: 0.926941) Vit-B_16, smooth=0.05
5. (Acc: 0.926   ) Vit-B_16, smooth=0.07, CenterCrop (augmentation)
6. (Acc: 0.9155  ) Vit-B_32, smooth=0.07
7. (Acc: ) ViT-L_16.npz, smooth=0.07
> CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py  --split overlap --num_steps 10000 --fp16 --name ViT-L_16_smooth0.07_bsize1 --smoothing_value 0.07 --model_type ViT-L_16 --pretrained_dir pretrained_models/ViT-L_16.npz --eval_batch_size 4 --train_batch_size 1 --output_dir output_L16

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
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py  --split overlap --num_steps 10000 --fp16 --name sample_smooth0.07 --smoothing_value 0.07 --model_type ViT-B_16 --pretrained_dir pretrained_models/ViT-B_16.npz 
```

## Reference
[*TransFG: A Transformer Architecture for Fine-grained Recognition (AAAI2022)*](https://arxiv.org/abs/2103.07976)  


