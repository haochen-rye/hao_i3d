# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str)
parser.add_argument('--modality', default='RGB', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet50")
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--warmup',  default=1, type=int,
                     help='using warm up strategy')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=30, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')

parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')

parser.add_argument('--fuse', default=False, action="store_true", help='use temporal fusion for models')
parser.add_argument('--fuse_group', default=1, type=int,  help='group number for temporal fusion')
parser.add_argument('--fuse_layer', default=['3.03','4.0'], type=str, nargs="+",
    help='the block place to add fusion module')
parser.add_argument('--fuse_dilation', default=False, action="store_true", help='use temporal fusion with dilation')
parser.add_argument('--fuse_spatial_dilation', default=1, type=int,  help='spatial dilation for temporal fusion')
parser.add_argument('--fuse_correlation', default=False, action="store_true", help='use temporal fusion with correlation')
parser.add_argument('--fuse_ave', default=False, action="store_true", help='use average for temporal fusion')
parser.add_argument('--fuse_downsample', default=False, action="store_true", help='downsample for temporal fusion')
parser.add_argument('--correlation_neighbor', default=3, type=int,  help='number for correlation neighbors')
parser.add_argument('--fuse_GroupConv', default=False, action="store_true", help='using group convolution for temporal fusion')

parser.add_argument('--spatial_dropout', default=False, action="store_true", help='using spatial dropout')
parser.add_argument('--sigmoid_thres', default=0.25, type=float, help='sigmoid thres')
parser.add_argument('--sigmoid_group', default=1, type=int,  help='group number for spatial dropout')
parser.add_argument('--sigmoid_random', default=False, action="store_true", help='using noise for sigmoid')
parser.add_argument('--sigmoid_layer', default=['3.0','4.0'], type=str, nargs="+",
    help='the block place to add sigmoid module')
parser.add_argument('--time3d', default=False, action="store_true", help='using 3d dropout for time')

parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')

parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
