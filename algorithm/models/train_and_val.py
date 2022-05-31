import sys

import torch
from timm.utils import AverageMeter
from tqdm import tqdm


def train(epoch, train_loader, model, criterion, optimizer, use_cuda):
    losses = AverageMeter()
    model.train()
    train_loader = tqdm(train_loader, file=sys.stdout, ncols=150, colour='MAGENTA')

    for i, (batch_x, batch_y) in enumerate(train_loader):
        input_var = batch_x.cuda(device=use_cuda)
        target_var = batch_y.cuda(device=use_cuda)

        out = model(input_var)
        loss = criterion(out, target_var)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loader.desc = "[train epoch {}]  lr:{}  loss:{}  ".format(epoch, optimizer.param_groups[-1]["lr"],
                                                                        round(losses.avg, 10))

    return losses.avg


def val(epoch, val_loader, model, criterion, use_cuda):
    losses = AverageMeter()
    model.eval()

    val_loader = tqdm(val_loader, file=sys.stdout, ncols=150, colour='CYAN')

    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(val_loader):
            input_var = batch_x.cuda(device=use_cuda)
            target_var = batch_y.cuda(device=use_cuda)

            out = model(input_var)

            loss = criterion(out, target_var)
            losses.update(loss.item())
            val_loader.desc = "[val epoch {}] loss:{}  ".format(epoch, round(losses.avg, 10))

    return losses.avg
