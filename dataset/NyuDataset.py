# coding=utf-8
import torch.utils.data as data
import sys
sys.path.append("..")
from imageio import imread
import numpy as np
import random
import time
from skimage.transform import resize
import glob
import os
import natsort
import matplotlib.pyplot as plt
import torch
from utils.utils import vis_data

def load_as_float(path):
    image = imread(path).astype(np.float32)[:, :, :3]
    return image

def load_as_npy(path):
    depth = np.load(path, dtype=np.float32)
    return depth

def get_inverse_depth(depth):
    return 1. / depth

def get_depth_mask(depth):
    return np.where(depth>0)

def get_inv_and_mask(depth:torch.Tensor):
    mask = depth > 0
    depth = torch.clamp(depth, 0.1, 10.)
    
    inv_depth = torch.zeros_like(depth)  # 모든 요소를 0으로 초기화
    inv_depth[mask] = 1. / depth[mask]  # mask가 True인 위치에서만 1/depth 계산
    
    return inv_depth, mask

class DataSequence(data.Dataset):
    def __init__(self, root,
                 seed=None,
                 train=True,
                 shuffle=True,
                 transform=None,
                 scene=None,
                 image_width=640,
                 image_height=480):
        np.random.seed(seed)
        random.seed(seed)
        self.root = root
        self.transform = transform
        # self.train_path = './nyu_depth_raw/train/'
        # self.valid_path = './nyu_depth_raw/validation/'
        
        self.shuffle = shuffle
        self.crawl_folders(scene=scene)

    def crawl_folders(self, scene):
        samples = []

        path = os.path.join(self.root, scene)
        imgs = glob.glob(path+'/rgb/*')
        depth = glob.glob(f'{path}/depth/*.npy')

        imgs = natsort.natsorted(imgs, reverse=False)
        depth = natsort.natsorted(depth, reverse=False)
        
        for i in range(len(imgs)):
            samples.append([imgs[i], depth[i]])

        if self.shuffle:
            random.shuffle(samples)
        self.samples = samples

    def __getitem__(self, index):
        img_path, depth_path = self.samples[index]

        imgs = imread(img_path).astype(np.float32)
        raw_depth = np.load(depth_path).astype(np.float32)

        raw_depth = np.expand_dims(raw_depth, axis=-1)
        mask = (raw_depth > 0) & (raw_depth < 10)

        zero_depth_mask = raw_depth == 0
        depth = np.zeros_like(raw_depth)  # 모든 값을 0으로 초기화
        non_zero_mask = ~zero_depth_mask  # 깊이가 0이 아닌 위치
        depth[non_zero_mask] = 1. / raw_depth[non_zero_mask]


        # vis_data(img=imgs, depth=raw_depth, mask=depth)
        if self.transform is not None:
            imgs, depth, mask = self.transform(imgs, depth, mask) # Imu (5, 11, 6) # intrinsic (3, 3)
        
        return imgs, depth, mask

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    start_time = time.time()
    D = DataSequence(
        root='./data/nyu_depth',
        shuffle=False,
        image_width=832,
        image_height=256,
        scene='train'
    )
    for img, depth, mask in D:
        print('time used {}'.format(time.time()-start_time))