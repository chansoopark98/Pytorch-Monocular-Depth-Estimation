import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import cv2

data_dir= '/home/park-ubuntu/park/Deep-Visual-Inertial-Odometry/depth/datasets'
nyu_train = tfds.load(name='nyu_depth_v2', data_dir=data_dir, split='train')
nyu_valid = tfds.load(name='nyu_depth_v2', data_dir=data_dir, split='validation')

train_path = './data/nyu_depth/train/'
valid_path = './data/nyu_depth/validation/'
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(os.path.join(train_path, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(train_path, 'depth'), exist_ok=True)
os.makedirs(os.path.join(valid_path, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(valid_path, 'depth'), exist_ok=True)


i = 0
for sample in nyu_train:
    print(i)
    
    image = sample['image'].numpy()[15:472, 7:630]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)

    depth = sample['depth'].numpy()[15:472, 7:630]
    depth = cv2.resize(depth, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(f'{train_path}/rgb/{str(i).zfill(6)}.png', image)
    np.save(f'{train_path}/depth/{str(i).zfill(6)}.npy', depth)

    i+=1

i = 0
for sample in nyu_valid:
    print(i)
    
    image = sample['image'].numpy()[15:472, 7:630]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    
    
    depth = sample['depth'].numpy()[15:472, 7:630]
    depth = cv2.resize(depth, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(f'{valid_path}/rgb/{str(i).zfill(6)}.png', image)
    np.save(f'{valid_path}/depth/{str(i).zfill(6)}.npy', depth)

    i+=1

