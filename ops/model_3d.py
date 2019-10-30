# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
from ops import resnet_3d

class i3d(nn.Module):
    def __init__(self, num_class, num_segments, 
                 base_model='resnet50', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 spatial_dropout=False, sigmoid_thres=0.2, sigmoid_group=1,
                 sigmoid_random=False, sigmoid_layer=['3.0','4.0'],
                 crop_num=1, print_spec=True, pretrain='imagenet'):
        super(i3d, self).__init__()
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain
        self.base_model_name = base_model
        self.new_length = 1
        self.spatial_dropout = spatial_dropout
        self.sigmoid_thres = sigmoid_thres
        self.sigmoid_group = sigmoid_group
        self.sigmoid_random = sigmoid_random
        self.sigmoid_layer = sigmoid_layer

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        num_segments:       {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.num_segments, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)
        # self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, 'fc').in_features
        if self.dropout == 0:
            setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, 'fc', nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, 'fc').weight, 0, std)
            constant_(getattr(self.base_model, 'fc').bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            if self.pretrain=='scratch':
                self.base_model = getattr(resnet_3d, base_model)(False)
            elif self.pretrain=='imagenet':
                self.base_model = getattr(resnet_3d, base_model)(True)
            elif self.pretrain=='kinetics':
                self.base_model = getattr(resnet_3d, base_model)(False)
                weight_pth = '/vulcan/scratch/hao/video/tsm/kinetic_res50.pt'
                weight = torch.load(weight_pth)
                self.base_model.state_dict().update(weight)
            else:
                raise NotImplementedError('no such pretrained mode' + self.pretrain)
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.spatial_dropout:
                from ops.spatial_dropout import make_spatial_dropout
                make_spatial_dropout(self.base_model, sigmoid_layer=self.sigmoid_layer,
                     thres=self.sigmoid_thres,  group=self.sigmoid_group, 
                     sigmoid_random=self.sigmoid_random)

    def forward(self, input):
        base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        return base_out

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                    GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
