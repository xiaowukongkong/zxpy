import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from algorithm.models.utils import read_image, normalization


# 只用于preprocess生成4*200*200的数据集
class Mydata_from_dng(Dataset):
    def __init__(self, path):
        self.path = path
        self.noises = os.listdir(os.path.join(self.path, 'noisy'))
        self.ground_truthes = os.listdir(os.path.join(self.path, 'ground truth'))
        self.noises = [n for n in self.noises if '.dng' in n]
        self.ground_truthes = [n for n in self.ground_truthes if '.dng' in n]
        self.noises = sorted(self.noises)
        self.ground_truthes = sorted(self.ground_truthes)

    def __len__(self):
        return len(self.noises)

    def __getitem__(self, index):
        n = os.path.join(self.path, 'noisy', self.noises[index])
        gt = os.path.join(self.path, 'ground truth', self.ground_truthes[index])

        # 读取两张图片，gt作为标签，n经过标准化作为原始输入
        n, height, width = read_image(n)
        n_normal = normalization(n, 1024, 16383)
        gt, _, _ = read_image(gt)
        gt_normal = normalization(gt, 1024, 16383)

        n_normal_expand = torch.from_numpy(np.transpose(
            n_normal.reshape(height // 2, width // 2, 4), (2, 0, 1))).float()
        gt_normal_expand = torch.from_numpy(np.transpose(
            gt_normal.reshape(height // 2, width // 2, 4), (2, 0, 1))).float()

        return n_normal_expand, gt_normal_expand


# 训练时用的dataset
class DataLoaderTrain(Dataset):
    def __init__(self, path, ps):
        super(DataLoaderTrain, self).__init__()
        self.path = path
        self.datas = os.listdir(self.path)
        self.ps = ps

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        index_ = index % len(self.datas)
        ps = self.ps

        [inp_img, tar_img] = np.load(os.path.join(self.path, self.datas[index_]))
        inp_img, tar_img = torch.from_numpy(inp_img), torch.from_numpy(tar_img)

        [_, w, h] = tar_img.shape
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(-1)
            tar_img = tar_img.flip(-1)
        elif aug == 2:
            inp_img = inp_img.flip(-2)
            tar_img = tar_img.flip(-2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(-1, -2))
            tar_img = torch.rot90(tar_img, dims=(-1, -2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(-1, -2), k=2)
            tar_img = torch.rot90(tar_img, dims=(-1, -2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(-1, -2), k=3)
            tar_img = torch.rot90(tar_img, dims=(-1, -2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(-1), dims=(-1, -2))
            tar_img = torch.rot90(tar_img.flip(-1), dims=(-1, -2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(-2), dims=(-1, -2))
            tar_img = torch.rot90(tar_img.flip(-2), dims=(-1, -2))

        return inp_img, tar_img
