python main_hao.py kinetics_400 --npb  --eval-freq=1 \
   --dropout 0.5  --num_segments 8  --dense_sample --gpus 4 -j 16 \
    --pretrain imagenet --lr_steps 30 45 54 --epochs 60 \
    --lr 0.02 -b 128 --arch resnet50   --warmup 2 --wd 5e-4 \
    --resume checkpoint/I3D_kinetics_400_resnet50_batch128_wd0.0005_avg_segment8_e75_dropout0.5_imagenet_lr0.02__warmup2_step48_64_72_dense/ckpt.pth.tar
    --spatial_dropout --sigmoid_layer 2.0 3.0 4.0 --sigmoid_thres 0.1


python main_hao.py kinetics_400 --npb --eval-freq=1 --dropout 0.5 --num_segments 8 --dense_sample --gpus 4 -j 16 --pretrain imagenet --lr_steps 32 48 56 --epochs 60 --lr 0.02 -b 128 --arch resnet50 --warmup 2 --wd 5e-4 --spatial_dropout --sigmoid_layer 2.0 3.0 4.0 --sigmoid_thres 0.1 \
--resume checkpoint/I3D_kinetics_400_resnet50_batch128_wd0.0005_avg_segment8_e60_dropout0.5_imagenet_lr0.02__warmup2_step32_48_56_dense_spatial_drop3d_0.1_group1_layer2.0_3.0_4.0/ckpt.pth.tar