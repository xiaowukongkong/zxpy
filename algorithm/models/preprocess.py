import os

import numpy as np
from torch.utils.data import DataLoader

from algorithm.models.dataset import Mydata_from_dng

height = 200
width = height

data = Mydata_from_dng(os.path.join('/usr/data/gamedata', 'dataset'))
train_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8)
src = os.path.join(os.getcwd(), f'dataset{height}')


def deldir(dir):
    """
    递归删除目录下面的所有文件
    :param dir:
    :return:
    """
    if not os.path.exists(dir):
        return False
    if os.path.isfile(dir):
        os.remove(dir)
        return
    for i in os.listdir(dir):
        t = os.path.join(dir, i)
        if os.path.isdir(t):
            deldir(t)
        else:
            os.unlink(t)
    os.removedirs(dir)


def process():
    i = 0
    if not os.path.exists(src):
        os.mkdir(src)
    else:
        deldir(src)

    for batch_x, batch_y in train_loader:
        [_, _, H, W] = batch_x.shape

        for h in range(batch_x.shape[2] // int(height)):
            for w in range(batch_x.shape[3] // int(width)):
                if (h + 1) * height + 50 > H or (w + 1) * width > W:
                    break
                x1 = batch_x[0, :, h * height + 50:(h + 1) * height + 50, W - (w + 1) * width:W - w * width].numpy()
                y1 = batch_y[0, :, h * height + 50:(h + 1) * height + 50, W - (w + 1) * width:W - w * width].numpy()
                if x1.shape[1] == height and x1.shape[2] == width:
                    np.save(os.path.join(src, f"data{i}.npy"), [x1, y1])
                    i += 1
                    print(i)


process()
