
# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

ROOT_DATASET = '/vulcan/scratch/hao/data'  


def return_ucf101(modality):
    filename_categories = 'UCF101/labels/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_rgb_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'sthv1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET 
        filename_imglist_train = 'sthv1/train_videofolder.txt'
        filename_imglist_val = 'sthv1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'sthv1/20bn-something-something-v1-flow'
        filename_imglist_train = 'sthv1/train_videofolder_flow.txt'
        filename_imglist_val = 'sthv1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_orig_something(modality):
    filename_categories = 174
    if modality == 'RGB':
        root_data = ROOT_DATASET 
        filename_imglist_train = 'sthv1/orig_train_videofolder.txt'
        filename_imglist_val = 'sthv1/orig_val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'sthv1/20bn-something-something-v1-flow'
        filename_imglist_train = 'sthv1/train_videofolder_flow.txt'
        filename_imglist_val = 'sthv1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_charades(modality):
    filename_categories = 157
    if modality == 'RGB':
        root_data = ROOT_DATASET 
        filename_imglist_train = 'charades/annotation/train_video.txt'
        filename_imglist_val = 'charades/annotation/val_video.txt'
        prefix = '{:06d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_diving(modality):
    filename_categories = 48
    if modality == 'RGB':
        root_data = ROOT_DATASET 
        filename_imglist_train = 'diving/train_list.txt'
        filename_imglist_val = 'diving/val_list.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_kinetics_200(modality):
    filename_categories = 200
    if modality == 'RGB':
        root_data = os.path.join(ROOT_DATASET, 'kinetics')
        filename_imglist_train = 'kinetics/annotation/kinetics_200_train.txt'
        filename_imglist_val = 'kinetics/annotation/kinetics_200_val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_kinetics_400(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = os.path.join(ROOT_DATASET, 'kinetics')
        filename_imglist_train = 'kinetics/annotation/kinetics_400_train.txt'
        filename_imglist_val = 'kinetics/annotation/kinetics_400_val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something, 
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51, 'diving': return_diving,
                   'kinetics_200': return_kinetics_200, 'kinetics_400': return_kinetics_400, 
                   'charades': return_charades, 'orig_something': return_orig_something}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
