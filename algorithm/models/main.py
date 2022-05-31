import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from algorithm.models.dataset import DataLoaderTrain
from algorithm.models.network import Restormer
from algorithm.models.train_and_val import train, val

lr = 1e-6
epochs = 200
save_dir = os.getcwd()

bs = {"model": 8}

'''
加载模型
'''
model_name = "model"

model_path = model_name + ".pth"
model_path = os.path.join(save_dir, model_path)

model = Restormer()
print("restormer")

use_cuda = 0
model.cuda(device=use_cuda)

'''
加载数据集
'''
ps = 192
dataset_path = os.path.join(os.getcwd(), 'dataset200')
full_dataset = DataLoaderTrain(dataset_path, ps=ps)

train_size = int(.95 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size],
                                                           generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=bs[model_name], shuffle=True, num_workers=12, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=bs[model_name], shuffle=False, num_workers=12)

if os.path.exists(model_path):
    model_info = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(use_cuda))
    print('==> loading existing model:', model_path)
    model.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=4, min_lr=1e-6,
                                                           eps=1e-8)
    cur_epoch = model_info['epoch']
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=4, min_lr=1e-6,
                                                           eps=1e-8)
    cur_epoch = 0

criterion1 = torch.nn.L1Loss()
criterion2 = torch.nn.MSELoss()
criterion = criterion1
criterion.cuda(device=use_cuda)

writer = {"model": SummaryWriter(f'./runs/l1_{ps}')}

for epoch in range(cur_epoch, epochs + 1):
    train_loss = train(epoch, train_loader, model, criterion, optimizer, use_cuda)

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict()},
        os.path.join(save_dir, model_path))

    val_loss = val(epoch, val_loader, model, criterion, use_cuda)
    scheduler.step(val_loss)

    writer[model_name].add_scalar('train_loss', train_loss, global_step=epoch)
    writer[model_name].add_scalar('val_loss', val_loss, global_step=epoch)
    writer[model_name].add_scalar('lr', optimizer.param_groups[-1]["lr"], global_step=epoch)
