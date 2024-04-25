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
import cv2

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

def vis_data(img, depth, mask):
    # 그리드 생성: 1행 3열
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 첫 번째 이미지 표시
    axes[0].imshow(img/255.)
    axes[0].set_title('img')
    axes[0].axis('off')  # 축 레이블 끄기

    # 두 번째 이미지 표시
    axes[1].imshow(depth)
    axes[1].set_title('depth')
    axes[1].axis('off')

    # 세 번째 이미지 표시
    axes[2].imshow(mask)
    axes[2].set_title('mask')
    axes[2].axis('off')

    # 전체 플롯 표시
    plt.show()


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
        # self.root = '/media/park-ubuntu/park_file/dataset/diode_depth/'
        self.transform = transform
        # self.train_path = './nyu_depth_raw/train/'
        # self.valid_path = './nyu_depth_raw/validation/'
        
        self.shuffle = shuffle
        self.crawl_folders(scene=scene)

    def crawl_folders(self, scene:str):
        samples = []

        path = os.path.join(self.root, scene)

        imgs = glob.glob(path + '/**/*.png', recursive=True)
        depths = glob.glob(path + '/**/*_depth.npy', recursive=True)
        masks = glob.glob(path + '/**/*_depth_mask.npy', recursive=True)
        
        imgs = natsort.natsorted(imgs, reverse=False)
        depths = natsort.natsorted(depths, reverse=False)
        masks = natsort.natsorted(masks, reverse=False)
        
        print(len(imgs))
        print(len(depths))
        print(len(masks))
        for i in range(len(imgs)):
            samples.append([imgs[i], depths[i], masks[i]])

        if self.shuffle:
            random.shuffle(samples)
        self.samples = samples

    def __getitem__(self, index):
        img_path, depth_path, mask_path = self.samples[index]

        imgs = imread(img_path).astype(np.float32)

        depth = np.load(depth_path).astype(np.float32)
        min_depth = 0.5
        max_depth = 10.
        
        depth = np.clip(depth, min_depth, max_depth)

        mask = np.load(mask_path).astype(np.float32)

        imgs = cv2.resize(imgs, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)
        
        depth = np.expand_dims(depth, axis=-1)
        
        range_mask = (depth > 0) & (depth < 10)
        
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(np.bool_)

        # zero_depth_mask = raw_depth == 0
        # depth = np.zeros_like(raw_depth)  # 모든 값을 0으로 초기화
        # non_zero_mask = ~zero_depth_mask  # 깊이가 0이 아닌 위치
        depth[mask] = 1. / depth[mask]

        depth[range_mask] = depth[range_mask]
        depth[depth >= 10] = 0

        # if np.max(depth) > 100:
        # vis_data(img=imgs, depth=depth, mask=np.float32(mask))

        if self.transform is not None:
            imgs, depth, mask = self.transform(imgs, depth, mask) # Imu (5, 11, 6) # intrinsic (3, 3)
        
        return imgs, depth, mask

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    start_time = time.time()
    D = DataSequence(
        root='/media/park-ubuntu/park_file/dataset/diode_depth/',
        shuffle=False,
        image_width=832,
        image_height=256,
        scene='train'
    )
    for img, depth, mask in D:
        print('time used {}'.format(time.time()-start_time))
