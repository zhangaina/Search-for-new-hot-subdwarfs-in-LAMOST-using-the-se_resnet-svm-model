import argparse
import random

import pandas as pd
from sklearn.metrics import confusion_matrix

import numpy as np
import time
from astropy.io import fits
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import os
from torch.utils import data
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,recall_score,precision_score
from imblearn.over_sampling import SMOTE
import Model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import math
from sklearn.metrics import confusion_matrix

# 使用sklearn调用衡量线性回归的MSE 、 RMSE、 MAE、r2
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def cutflux(wave, flux):
    returnflux = []
    for i in range(len(wave)):
        if wave[i] >= 3800.0 and wave[i] <= 8000.0:
           returnflux.append(flux[i])
        # elif wave[i] >= 5350.0 and wave[i] <= 5450.0:
        #     returnflux.append(flux[i])
        # elif wave[i] >= 6500.0 and wave[i] <= 6640.0:
        #     returnflux.append(flux[i])
        else:
            continue
    return returnflux

class MyDataset(data.Dataset):
    def __init__(self,datas,obsids):
        self.Data = np.asarray(datas)  # 数据集
        self.Obsid = np.asarray(obsids)
    def __getitem__(self, index):
        # 把numpy转换为Tensor
        data = torch.from_numpy(self.Data[index]).reshape(1,3909)#1142#torch.from_numpy(self.Data[index]).reshape(1,1213)
        obsid = torch.tensor(self.Obsid[index])
        return data,obsid

    def __len__(self):
        return len(self.Data)


def test(model):
    model.eval()

    predictionLog = []
    obsidList = []  # 修改这里，将obsid改为obsidList或其他名称，避免与变量名重复

    batch = tqdm(test_dataloader)
    with torch.no_grad():
        for inputs, obsid in batch:
            inputs = inputs.type(torch.FloatTensor).cuda()
            obsid_tensor = obsid  # 使用新的变量名来存储Tensor
            obsid = obsid_tensor.type(torch.FloatTensor).cuda()
            outputs = model(inputs)

            for i in outputs:
                predictionLog.append(i.item())
            for i in obsid_tensor:  # 使用新的变量名
                obsidList.append(i.item())  # 将obsid添加到obsidList中

    # 构建字典
    point = {'obsid': obsidList, 'teff': predictionLog}
    print(point)
    print(len(point))

    # 保存为CSV文件
    pd.DataFrame(point).to_csv('/home/admin1/data1/czd/pred/176logy.csv')



def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train_and_test')
    parser.add_argument('--batch_size', type=int, default=8 ,help='batch size')
    parser.add_argument('--epochs', type=int, default=700, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='epochs')
    args = parser.parse_args()


    all_x=[]
    obsid = []

    model = Model.Get_se_resnet_19_1d(num_classes=1)
    model.load_state_dict(torch.load('/home/admin1/data1/czd/pred/logy.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    hdu = fits.open('/home/admin1/data1/czd/pred/spec176.fits')
    for i in range(len(hdu[1].data)) :
        all_x.append(hdu[1].data[i]['flux'])
        obsid.append(hdu[1].data[i]['obsid'])
    test_dataset = MyDataset(all_x,obsid)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    test(model)


