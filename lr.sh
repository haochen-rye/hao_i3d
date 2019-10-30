python main_hao.py orig_something  --npb  --eval-freq=1 \
   --dropout 0.5  --num_segments 8  --gpus 4 -j 16 \
    --pretrain imagenet --lr_steps 24 32  --epochs 36 \
    --lr 0.02 -b 124 --arch resnet50   --warmup 2 --wd 5e-4 \
    --spatial_dropout --sigmoid_layer 2.02 3.024 4.02 --sigmoid_thres 0.15
python main_hao.py orig_something  --npb  --eval-freq=1 \
   --dropout 0.5  --num_segments 8  --gpus 4 -j 16 \
    --pretrain imagenet --lr_steps 27 34  --epochs 40 \
    --lr 0.02 -b 128 --arch resnet50   --warmup 2 --wd 5e-4 \
    --spatial_dropout --sigmoid_layer 3.0 --sigmoid_thres 0.75 \
    --resume checkpoint/I3D_orig_something_resnet50_batch124_wd0.0005_avg_segment8_e36_dropout0.5_imagenet_lr0.02__warmup2_step24_32_spatial_drop3d_0.75_group1_layer3.0/ckpt.pth.tar