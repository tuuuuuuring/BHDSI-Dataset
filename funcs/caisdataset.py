import logging
import os
import csv
import pandas as pd
import random
import numpy as np
import argparse
from PIL import Image
import cv2
import torchvision.transforms as T
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, ConcatDataset


def count_files_in_directory(directory_path):
    """
    统计给定目录下的文件数量。

    参数:
    directory_path (str): 目录路径

    返回:
    int: 目录下的文件数量
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} does not exist.")

    # 统计文件数量
    file_count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
    return file_count


def generate_string_array(cofp):
    """
    生成一个格式化字符串的数组，形如 ['0001', '0002', ..., 'cofp']

    参数:
    cofp (int): 数组的最大整数值

    返回:
    list: 格式化字符串的数组
    """
    # 确定格式化字符串的宽度
    width = len(str(1000))

    # 生成格式化字符串数组
    formatted_strings = [f'{i:0{width}d}' for i in range(1, cofp + 1)]

    return formatted_strings


def read_data(index_file, img_dir, uis_name='cbra', data_name='default'):

    # data_frame = pd.read_csv(index_file, encoding="utf-8")

    images_b2 = []
    images_b3 = []
    images_b4 = []
    images_b8 = []
    images_vv = []
    images_vh = []
    masks_bh = []
    masks_uis = []
    images_name = []
    tocount=os.path.join(img_dir,'_bhr')
    cofp=count_files_in_directory(tocount)
    namearray=generate_string_array(cofp)
    cut=os.path.basename(os.path.normpath(img_dir))

    # for index, row in data_frame.iterrows():
    #     city_name, tile_name = row['city_name'], row['tile_id']
    #
    #     b2 = os.path.join(img_dir, city_name + '_b2', tile_name)
    #     b3 = os.path.join(img_dir, city_name + '_b3', tile_name)
    #     b4 = os.path.join(img_dir, city_name + '_b4', tile_name)
    #     b8 = os.path.join(img_dir, city_name + '_b8', tile_name)
    #     vv = os.path.join(img_dir, city_name + '_VV', tile_name)
    #     vh = os.path.join(img_dir, city_name + '_VH', tile_name)
    #     bh = os.path.join(img_dir, city_name + '_height', tile_name)
    #
    #     uis_cbra = os.path.join(img_dir, city_name + '_cbra256', tile_name)
    #     uis_wsf = os.path.join(img_dir, city_name + '_wsf', tile_name)
    #
    #     if city_name == 'changzhou':
    #         continue
    #
    #     if uis_name == 'cbra':
    #         uis = uis_cbra
    #     elif uis_name == 'wsf':
    #         uis = uis_wsf
    #     else:
    #         raise Exception('建筑物轮廓数据不在列表！')
    #
    #     images_b2.append(b2)
    #     images_b3.append(b3)
    #     images_b4.append(b4)
    #     images_b8.append(b8)
    #     images_vv.append(vv)
    #     images_vh.append(vh)
    #     masks_bh.append(bh)
    #     masks_uis.append(uis)
    #     images_name.append(city_name + '_' + tile_name)

    for i in range(cofp):
        b2=os.path.join(img_dir,'sentinel2_AREA_b2',namearray[i]+'_b2.tif')
        b3 = os.path.join(img_dir, 'sentinel2_AREA_b3', namearray[i] + '_b3.tif')
        b4 = os.path.join(img_dir, 'sentinel2_AREA_b4', namearray[i] + '_b4.tif')
        b8 = os.path.join(img_dir, 'sentinel2_AREA_b8', namearray[i] + '_b8.tif')
        vv = os.path.join(img_dir, 'sentinel1_AREA_vv', namearray[i] + '_vv.tif')
        vh = os.path.join(img_dir, 'sentinel1_AREA_vh', namearray[i] + '_vh.tif')
        bh = os.path.join(img_dir, '_bhrf', namearray[i] + '.tif')
        uis = os.path.join(img_dir, 'cbra2018', namearray[i] + '.tif')
        images_b2.append(b2)
        images_b3.append(b3)
        images_b4.append(b4)
        images_b8.append(b8)
        images_vv.append(vv)
        images_vh.append(vh)
        masks_bh.append(bh)
        masks_uis.append(uis)
        images_name.append(cut + '_' + namearray[i]+'.tif')
    return images_name, images_b2, images_b3, images_b4, images_b8, images_vv, images_vh, masks_bh, masks_uis


class RSDataset(Dataset):
    def __init__(self, args, index_file, img_dir):

        images_name, images_b2, images_b3, images_b4, images_b8, images_vv, images_vh, masks_bh, masks_uis = \
            read_data(index_file, img_dir, uis_name=args.uis_name, data_name=args.data_name)

        self.images_b2 = images_b2
        self.images_b3 = images_b3
        self.images_b4 = images_b4
        self.images_b8 = images_b8
        self.images_vv = images_vv
        self.images_vh = images_vh
        self.masks_bh = masks_bh
        self.masks_uis = masks_uis
        self.images_name = images_name

        self.args = args
        self.img_dir = img_dir
        # self.transform = transform
        self.phase = args.phase

        logging.info(f'Creating dataset with {len(masks_uis)} examples')

    def __len__(self):
        return len(self.masks_uis)

    def __getitem__(self, i):

        img_name = self.images_name[i]
        img_b2, img_b3, img_b4, img_b8, img_vv, img_vh, img_bh, img_uis = self.get_image(i)


        img_b2, img_b3, img_b4, img_b8, img_vv, img_vh, img_bh, img_uis = \
            self.preprocess(img_b2, img_b3, img_b4, img_b8, img_vv, img_vh, img_bh, img_uis)

        if self.args.s1_band == 3:
            s1_img = np.stack([img_vv, img_vh, img_b8], axis=-1)
        elif self.args.s1_band == 2:
            s1_img = np.stack([img_vv, img_vh], axis=-1)
        else:
            raise Exception('args.s1_band is not defined!')

        if self.args.s2_band == 4:
            s2_img = np.stack([img_b2, img_b3, img_b4, img_b8], axis=-1)
        elif self.args.s2_band == 3:
            s2_img = np.stack([img_b2, img_b3, img_b4], axis=-1)
        else:
            raise Exception('args.s2_band is not defined!')

        img_bh = torch.tensor(img_bh).unsqueeze(dim=0)
        img_uis = torch.tensor(img_uis).unsqueeze(dim=0)

        s1_tensor = torch.tensor(s1_img).permute(2,0,1)
        s2_tensor = torch.tensor(s2_img).permute(2,0,1)

        return s1_tensor, s2_tensor, img_bh, img_uis

    def get_image(self, idx: int):
        b2_path, b3_path, b4_path, b8_path, vv_path, vh_path, mask_bh_path, mask_uis_path = \
            self.images_b2[idx], self.images_b3[idx], self.images_b4[idx], \
                self.images_b8[idx], self.images_vv[idx], self.images_vh[idx],\
                self.masks_bh[idx], self.masks_uis[idx]

        img_b2 = Image.open(b2_path)
        img_b3 = Image.open(b3_path)
        img_b4 = Image.open(b4_path)
        img_b8 = Image.open(b8_path)
        img_vv = Image.open(vv_path)
        img_vh = Image.open(vh_path)
        target_bh = Image.open(mask_bh_path)
        target_uis = Image.open(mask_uis_path)

        img_b2 = img_b2.resize((256, 256), Image.Resampling.BILINEAR)
        img_b3 = img_b3.resize((256, 256), Image.Resampling.BILINEAR)
        img_b4 = img_b4.resize((256, 256), Image.Resampling.BILINEAR)
        img_b8 = img_b8.resize((256, 256), Image.Resampling.BILINEAR)
        img_vv = img_vv.resize((256, 256), Image.Resampling.BILINEAR)
        img_vh = img_vh.resize((256, 256), Image.Resampling.BILINEAR)
        target_bh = target_bh.resize((256, 256), Image.Resampling.NEAREST)
        target_uis = target_uis.resize((256, 256), Image.Resampling.NEAREST)

        # return img_b2, img_b3, img_b4, img_b8, img_vv, img_vh, target_bh, target_uis
        return np.array(img_b2).astype('float32'), \
            np.array(img_b3).astype('float32'), \
            np.array(img_b4).astype('float32'), \
            np.array(img_b8).astype('float32'), \
            np.array(img_vv).astype('float32'), \
            np.array(img_vh).astype('float32'), \
            np.array(target_bh).astype('float32'), \
            np.array(target_uis).astype('float32')


    def preprocess(self, img_b2, img_b3, img_b4, img_b8, img_vv, img_vh, img_bh, img_uis):

        ####只标准化
        maxb8= 1.4469000101089478
        minb8= 0.016200000420212746
        b8std= 0.06967487323859937
        meanb8= 0.15573851139390163
        maxb2= 1.1100000143051147
        minb2= 0.041999999433755875
        b2std= 0.02504830033552796
        meanb2= 0.13887002717847197
        maxb3= 1.228600025177002
        minb3= 0.029200000688433647
        b3std= 0.02702816873024951
        meanb3= 0.12237092260452842
        maxb4= 1.458299994468689
        minb4= 0.017500000074505806
        b4std= 0.04012980389187709
        meanb4= 0.1115794893766814
        maxvv= 39.737640380859375
        minvv= -50.0
        vvstd= 9.414333835144712
        meanvv= -15.40915998142092
        maxvh= 29.298410415649414
        minvh= -50.0
        vhstd= 9.993993388446894
        meanvh= -23.915761885939805

        vvstd = 5.894991953747706
        meanvv = -9.400159472318535
        vhstd = 6.5062821612476816
        meanvh = -17.48087407519381
        b2std = 0.041234904196493866
        meanb2 = 0.09032810650349908
        b3std = 0.044207560644641535
        meanb3 = 0.11008062839901452
        b4std = 0.058242708431953054
        meanb4 = 0.11641374186227577
        b8std = 0.07429693760337167
        meanb8 = 0.20418531308997234

        # 为什么像素值会出现负数呢？
        img_b2[img_b2 < 0] = 0
        # img_b2[img_b2>1.5]=1.5

        img_b3[img_b3 < 0] = 0
        # img_b3[img_b3>1.5]=1.5

        img_b4[img_b4 < 0] = 0
        # img_b4[img_b4>1.5]=1.5

        img_b8[img_b8 < 0] = 0
        # img_b8[img_b8>1.5]=1.5

        # img_vv[img_vv>100]=100
        img_vv[img_vv < -50] = -50

        # img_vh[img_vh>100]=100
        img_vh[img_vh < -50] = -50

        # 建筑物高度计算：楼层数 * 3（米/层）
        img_bh=img_bh/3
        img_bh[img_bh<0] = 0
        img_bh[img_bh == np.nan] = 0
        img5=img_bh*5
        img4=img_bh*3.1 # 原来是3.1
        img3=img_bh*3
        img_bh[np.where(img_bh>=30)]=img5[np.where(img_bh>=30)]
        img_bh[np.logical_and(img_bh>=20, img_bh<30)]=img4[np.logical_and(img_bh>=20, img_bh<30)]
        img_bh[np.where(img_bh<20)]=img3[np.where(img_bh<20)]
        img_bh[img_bh<3]=0


        # 建筑物轮廓数据处理
        img_uis[img_uis < 0] = 0
        img_uis[img_uis > 255] = 0
        # img_uis[img_uis > 0] = 1
        # img_uis[img_uis == 255] = 1
        # img_uis = np.resize(img_uis, (256, 256))

        img_b2 = (img_b2 - meanb2) / b2std
        img_b3 = (img_b3 - meanb3) / b3std
        img_b4 = (img_b4 - meanb4) / b4std
        img_b8 = (img_b8 - meanb8) / b8std
        img_vv = (img_vv - meanvv) / vvstd
        img_vh = (img_vh - meanvh) / vhstd

        return img_b2, img_b3, img_b4, img_b8, img_vv, img_vh, img_bh, img_uis

def get_dataset(args):

    data_path = args.data_path
    train_index = args.train_index
    val_index = args.val_index

    train_dataset = RSDataset(args, train_index, data_path)
    val_dataset = RSDataset(args, val_index, data_path)

    return train_dataset # , val_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Code for Building Height Estimate')
    args = parser.parse_args()

    # my data
    args.data_path = r'E:\Project\building_height\new_data\2_train_dataset_filter_0.1'
    args.train_index = '../dataloader/wsy_path/bh_uis_train_0.1.csv'
    args.val_index = '../dataloader/wsy_path/val_0.1.csv'

    args.data_path = r'D:\paisenPR\bhr_by_dl\data4\cut1' # r'/root/autodl-tmp/3_Training_tif'
    args.train_index = '../data_path/ntraining_data_cbw_30.csv' # '../dataloader/cai_path/ntraining_data_cbw_30.csv'
    args.val_index = '../data_path/nvalidation_data_cbw.csv' # '../dataloader/cai_path/nvalidation_data_cbw.csv'

    args.s1_band = 2
    args.s2_band = 4
    args.phase = 'test'
    args.expand_index = 'None'
    args.uis_name = 'wsf'
    args.data_name = 'cai'

    # train_dataset, val_dataset = get_dataset(args)
    train_datasets=[]
    for i in range(67):
        args.data_path = os.path.join(r'D:\paisenPR\bhr_by_dl\data4', f'cut{i+1}')
        train_dataset = get_dataset(args)
        train_datasets.append(train_dataset)
    print('done')
    all=ConcatDataset(train_datasets)
    print(' len train dataset', str(len(all)))
    # print(' len train dataset', str(len(train_dataset)))
    # print(' len val dataset', str(len(val_dataset)))

    # for i, data in enumerate(tqdm(all)):
    #     s1_img, s2_img, mask_bh, mask_uis, img_name = data
    #     bhmax=mask_bh.max()
    #     bhmin=mask_bh.min()
    #     uismax=mask_uis.max()
    #     uismin=mask_uis.min()
    #     mask=mask_uis==1
    #     numones=mask.sum().item()
    #     print(bhmin,bhmax,uismin,uismax,numones)

