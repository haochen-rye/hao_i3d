python main_charades.py charades  --npb  --eval-freq=1 \
   --dropout 0.2  --num_segments 64  --gpus 4 -j 16 \
    --pretrain kinetics --lr_steps 60 90 --epochs 100 \
    --lr 0.002 -b 16 --arch resnet50 --wd 5e-4 --warmup 1

python main_charades.py charades  --npb  --eval-freq=1 \
   --dropout 0.5  --num_segments 64  --gpus 4 -j 16 \
    --pretrain kinetics --lr_steps 30 45 --epochs 50 \
    --lr 0.01 -b 16 --arch resnet50   --warmup 1 --wd 5e-4 \
    --spatial_dropout --sigmoid_layer 3.0 --sigmoid_thres 0.15      
