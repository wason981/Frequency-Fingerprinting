import os
for id in range(100,1300,100):
    os.system(f"CUDA_VISIBLE_DEVICES=3 python train_v9.py --dataset=tiny_imagenet --epsilon=0.01 --fea_w=1  --perc_w=0 --test --train_batch_size=200 --load_ckpt={id}")