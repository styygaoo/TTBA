import torch
import numpy as np
from torchvision import transforms, utils
from PIL import Image

def unpack_and_move(data):
    if isinstance(data, (tuple, list)):
        image = data[0].to(device, non_blocking=True)
        gt = data[1].to(device, non_blocking=True)
        return image, gt
    if isinstance(data, dict):
        # print("hier")
        keys = data.keys()
        image = data['image'].to(device, non_blocking=True)
        gt = data['depth'].to(device, non_blocking=True)
        # print(image.shape)
        # print(gt.shape)
        return image, gt
    print('Type not supported')

def inverse_depth_norm(depth):
    depth = maxDepth / depth
    depth = torch.clamp(depth, maxDepth / 100, maxDepth)
    return depth

def depth_norm(self, depth):
    depth = torch.clamp(depth, maxDepth / 100, maxDepth)
    depth = maxDepth / depth
    return depth

class CenterCrop(object):
    """
    Wrap torch's CenterCrop
    """
    def __init__(self, output_resolution):
        print(output_resolution)
        self.crop = transforms.CenterCrop(output_resolution)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image))
        if isinstance(depth, np.ndarray):
            depth = Image.fromarray(depth)
        image = self.crop(image)
        depth = self.crop(depth)

        return {'image': image, 'depth': depth}


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
maxDepth = 80