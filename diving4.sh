python main_hao.py diving  --npb  --eval-freq=1 \
   --dropout 0.5  --num_segments 32  --gpus 4 -j 16 \
    --pretrain imagenet --lr_steps 18 23 --epochs 25 \
    --lr 0.01 -b 32 --arch resnet50   --warmup 2 --wd 5e-4 \
    --spatial_dropout --sigmoid_layer 2.0 --sigmoid_thres 0.15
python main_hao.py diving  --npb  --eval-freq=1 \
   --dropout 0.5  --num_segments 32  --gpus 4 -j 16 \
    --pretrain imagenet --lr_steps 18 24 --epochs 27 \
    --lr 0.01 -b 32 --arch resnet50   --warmup 2 --wd 5e-4 \
    --spatial_dropout --sigmoid_layer 3.0 4.0 --sigmoid_thres 0.1    
python main_hao.py diving  --npb  --eval-freq=1 \
   --dropout 0.5  --num_segments 32  --gpus 4 -j 16 \
    --pretrain imagenet --lr_steps 18 24 --epochs 27 \
    --lr 0.01 -b 32 --arch resnet50   --warmup 2 --wd 5e-4 \
    --spatial_dropout --sigmoid_layer 2.02 --sigmoid_thres 0.15