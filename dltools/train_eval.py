from dltools.config_common import opt
import os
import torch as t
from tqdm import tqdm
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def FCNetRegressTrain(model, optimizer, train_loader, train_data, test_data, **kwargs):
    opt._parse(kwargs)
    model.to(opt.device)
    train_data = train_data.to(opt.device)
    test_data = test_data.to(opt.device)

    criterion = opt.criterion

    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay)  # 设置学习率下降策略

    # train
    print("training on ", opt.device)

    train_mse = np.zeros(opt.max_epoch+1, dtype=np.float)
    test_mse = np.zeros(opt.max_epoch+1, dtype=np.float)

    train_mse[0] = val(model, train_data, criterion)
    test_mse[0] = val(model, test_data, criterion)

    for epoch in range(opt.max_epoch):

        train_loss_sum, n = 0.0, 0

        model.train()
        for ii, (x, y) in enumerate(train_loader):
            # forward
            input = x.to(opt.device)
            target = y.to(opt.device)
            score = model(input)

            # backward
            optimizer.zero_grad()
            loss = criterion(score, target)
            loss.backward()

            # update weights
            optimizer.step()

            train_loss_sum += loss.item() * len(target)
            n += target.shape[0]


            if (ii + 1) % opt.print_freq == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'
                      .format(epoch + 1, n, len(train_loader.dataset),
                              100. * n / len(train_loader.dataset), train_loss_sum / n)
                      )

        train_mse[epoch+1] = val(model, train_data, criterion)
        test_mse[epoch+1] = val(model, test_data, criterion)
        if (epoch + 1) % opt.epoch_freq == 0:
            print("epoch: {}\tlr: {}\ttrain_mse: {:.6f}\ttest_mse: {:.6f}".
                  format(epoch+1, scheduler.get_last_lr()[0], train_mse[epoch+1], test_mse[epoch+1])
            )
        scheduler.step()
    return train_mse, test_mse



@t.no_grad()
def val(model, data, criterion):

    model.eval()
    val_input = data[:,:(data.shape[1]-1)]
    val_target = data[:,-1]
    val_score = model(val_input)
    val_score = val_score.squeeze()
    loss = criterion(val_score, val_target)
    return loss

