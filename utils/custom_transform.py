from __future__ import division
import torch
import random
import numpy as np
# from scipy.misc import imresize
from skimage.transform import resize
from torchvision.transforms import Resize, ToPILImage, ToTensor as TorchToTensor

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, depth, mask):
        for t in self.transforms:
            image, depth, mask = t(image, depth, mask)
        return image, depth, mask

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, depth, mask):

        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return image, depth, mask
   
class ToTensor(object):
    def __call__(self, image, depth, mask):
        image = np.transpose(image, (2, 0, 1))
        depth = np.transpose(depth, (2, 0, 1))
        
        image = torch.from_numpy(image).float()/255.
        
        return image, depth, mask

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, image, depth, mask):
        
        if random.random() < 0.5:
            image = np.copy(np.fliplr(image))
            depth = np.copy(np.fliplr(depth))
            mask = np.copy(np.fliplr(mask))
        return image, depth, mask

class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, image, depth, mask):
        in_h, in_w, _ = image.shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        scaled_images = resize(image, (scaled_h, scaled_w))
        scaled_depths = resize(depth, (scaled_h, scaled_w), order=0, anti_aliasing=False)
        scaled_masks = resize(mask, (scaled_h, scaled_w), order=0, anti_aliasing=False)

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = scaled_images[offset_y:offset_y + in_h, offset_x:offset_x + in_w]
        cropped_depths = scaled_depths[offset_y:offset_y + in_h, offset_x:offset_x + in_w]
        cropped_masks = scaled_masks[offset_y:offset_y + in_h, offset_x:offset_x + in_w]
        return cropped_images, cropped_depths, cropped_masks

class AugmentImagePair(object):
    def __init__(self, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2]):
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, image, depth, mask):
        p = np.random.uniform(0, 1, 1)

        if p > 0.5:
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)
        
            # randomly shift gamma
            image = image ** random_gamma

            # randomly shift brightness
            image = image * random_brightness

            # randomly shift color
            for i in range(3):
                image[i, :, :] *= random_colors[i]

            # saturate
            image = torch.clamp(image, 0, 1)

        return image, depth, mask