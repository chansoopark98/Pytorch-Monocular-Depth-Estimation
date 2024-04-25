import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

sample_path = '/media/park-ubuntu/park_file/dataset/diode_depth/train'
# save_path = '/home/park/diode_converted/train/'
# os.makedirs(save_path, exist_ok=True)
# os.makedirs(save_path + 'image', exist_ok=True)
# os.makedirs(save_path + 'depth', exist_ok=True)

def depth_inpaint(depth, max_value=10, missing_value=0):
    depth = np.where(depth > 10, 0, depth)

    depth = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (depth == missing_value).astype(np.uint8)

    scale = np.abs(depth).max()
    depth = depth.astype(np.float32) / scale
    depth = cv2.inpaint(depth, mask, 1, cv2.INPAINT_NS)

    depth = depth[1:-1, 1:-1]
    depth = depth * scale

    return depth

def plot_depth_map(dm, validity_mask):
    validity_mask = validity_mask > 0
    MIN_DEPTH = 0.5
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    dm = np.log(dm, where=validity_mask)

    dm = np.ma.masked_where(~validity_mask, dm)

    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    plt.imshow(dm, cmap=cmap, vmax=np.log(MAX_DEPTH))
    plt.show()

print('sample_path', sample_path)
locations = glob.glob(str(sample_path) + '/*')
print('locations', locations)
idx = 0
# 실내/ 실외 구분
for location in locations:

    scenes = glob.glob(location + '/*')
    
    # Scene 구분
    for scene in tqdm(scenes, total=len(scenes)):
    # for scene in scenes:
        scans = glob.glob(scene + '/*')
        # Scane 구분
        for scan in tqdm(scans, total=len(scans)):
        # for scan in scans:
            data_list = glob.glob(scan + '/*.png')
            # Data 구분
            for data in data_list:
                image = cv2.imread(data, cv2.IMREAD_COLOR)

                # depth = '.' + data.split('.')[0] + '_depth.npy'
                depth = data.replace('.png', '_depth.npy')
                mask = data.replace('.png', '_depth_mask.npy')
                mask = np.load(mask)
                print(mask)

                depth = np.load(depth)

                plt.imshow(depth)
                plt.show()

                # depth = depth_inpaint(depth)

                plot_depth_map(dm=depth[:, :, 0], validity_mask=mask)


                depth = np.reshape(depth, [768, 1024]).astype(np.float32)

                idx += 1
                
                # cv2.imwrite(save_path + 'image/' + '_' + str(idx) + '.jpg', image)
                # np.save(save_path + 'depth/' + '_' + str(idx) + '.npy', depth)
                # cv2.imwrite(save_path + 'depth/' + '_' + str(idx) + '.png', depth)