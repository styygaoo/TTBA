import torch
from torch.utils.data import Dataset
import os
import numpy as np
from data.data_processing import CenterCrop


class VKITTI(Dataset):
    def __init__(self, path, resolution=(384, 1280)):  # Initialize with your dataset
        self.path = path
        self.resolution = resolution
        self.files = os.listdir(self.path)
        # print(self.files)
        # print(self.resolution)
        # self.trans = transforms.Compose([transforms.Resize(size=(384, 1280))])
        # self.downscale_image = transforms.Resize(self.resolution)  # To Model resolution
        self.transform = CenterCrop(self.resolution)

    def __len__(self):                  # upperbound for sample index
        return len(self.files)

    def __getitem__(self, index):

        image_path = os.path.join(self.path, self.files[index])

        data = np.load(image_path, allow_pickle=True)
        # depth, image = data['depth'], data['image']
        # depth = np.expand_dims(depth, axis=2)
        data = self.transform(data)
        image, depth = data['image'], data['depth']
        # print(image.shape)
        image = np.array(image)
        depth = np.array(depth)
        # print(image.shape)
        return image, depth


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
maxDepth = 80