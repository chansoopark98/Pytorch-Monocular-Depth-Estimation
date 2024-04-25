import numpy as np
import numpy
import torch
import shutil
from path import Path
import datetime
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import os
import random

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

def json_out(dictionary, outpath, outname):
	with open(os.path.join(outpath, outname), 'w', encoding="utf-8") as f_out:
		jsObj = json.dumps(dictionary, indent=4)
		f_out.write(jsObj)
		f_out.close()

def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    return save_path/timestamp

def save_checkpoint(save_path, dispnet_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['DepthNet']
    states = [dispnet_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path + '/{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path + '/{}_{}'.format(prefix,filename),
                            save_path + '/{}_best.pth.tar'.format(prefix))

def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0,1,low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0,max_value,resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:,i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)

def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )
    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    tensor[torch.isinf(tensor)] = 0
    tensor[torch.isnan(tensor)] = 0
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    return array


def tensor2array2(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    tensor[torch.isinf(tensor)] = 0
    tensor[torch.isnan(tensor)] = 0
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    else:
        raise ValueError('wrong dimention of tensor.shape {}'.format(tensor.shape))
    return (array.transpose(1,2,0) * 255).astype('uint8')

def tensor2img(tensor):
    img = tensor.cpu().data.numpy()
    mean = np.array([0.5, 0.5, 0.5])
    img = np.transpose(img, (0, 2, 3, 1))
    img = img*mean + mean
    return img

def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)