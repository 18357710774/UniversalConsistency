import torch
import math
import numpy as np
import random
import os


def Simf1TrainDataTensor(N=2000, d=3, c=1, noise=0.1, seed=None):
    if seed != None:
        torch.manual_seed(seed)
    x = torch.rand(N, d)
    e = noise * torch.randn(N, 1)
    y_pure = Simf1_torchTensor(x, c)
    y = y_pure + e
    return x, y


def Simf1TrainDataArray(N=2000, d=3, c=1, noise=0.1, seed=None):
    if seed != None:
        np.random.seed(seed)
    x = np.random.rand(N, d)
    e = noise * np.random.randn(N, 1)
    y_pure = Simf1_numpyArray(x, c)
    y = y_pure + e
    return x, y


def Simf1TestDataTensor(N=1000, d=3, c=1, seed=None):
    if seed != None:
        torch.manual_seed(seed+100)
    x = torch.rand(N, d)
    y = Simf1_torchTensor(x, c)
    return x, y


def Simf1TestDataArray(N=1000, d=3, c=1, seed=None):
    if seed != None:
        np.random.seed(seed+100)
    x = np.random.rand(N, d)
    y = Simf1_numpyArray(x, c)
    return x, y


def Simf1_torchTensor(x, c):
    N = x.shape[0]
    y1 = torch.zeros(N,1)
    r = torch.norm(x, p='fro', dim=1, keepdim=True) / c
    y2 = (1 - r) ** 6 * (35 * (r ** 2) + 18 * r + 3)
    y = torch.where(r <= 1, y2, y1)

    # y3 = torch.zeros(N,1)
    # rr = torch.norm(x, p='fro', dim=1) / c
    # for i in range(N):
    #     if rr[i] <= 1:
    #         y3[i] = (1 - rr[i]) ** 6 * (35 * (rr[i] ** 2) + 18 * rr[i] + 3)
    # dif = (y3-y).abs().max()
    return y


def Simf1_numpyArray(x, c):
    N = x.shape[0]
    y1 = np.zeros((N, 1))
    r = np.linalg.norm(x, ord=2, axis=1, keepdims=True) / c
    y2 = (1 - r) ** 6 * (35 * (r ** 2) + 18 * r + 3)
    y = np.where(r <= 1, y2, y1)
    return y


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)            # if you are using multi-GPU.



