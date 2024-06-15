import argparse
import random

# import torch.nn.functional as F   # 激励函数都在这

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
#import DenseNEt
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
from tensorboardX import SummaryWriter

def weight_mae_loss(y_pred, y_true, weight=3):
    loss = torch.abs(y_true - y_pred)
    condition = torch.logical_and(torch.greater(y_true, 39000), torch.less(y_true, 81000))
    weighted_loss = torch.where(condition, loss * weight, loss) #按一定的规则合并torch类型
    return torch.mean(weighted_loss)

# class HuberLoss(nn.Module):
#     def __init__(self, delta=1.0):
#         super(HuberLoss, self).__init__()
#         self.delta = delta  # delta 是 Huber Loss 的超参数，控制平滑区域的大小
#
#     def forward(self, y_pred, y_true):
#         residual = torch.abs(y_pred - y_true)
#         loss = torch.where(residual < self.delta, 0.5 * residual ** 2, self.delta * (residual - 0.5 * self.delta))
#         return torch.mean(loss)


def get_mae(y_hat, y):
    "计算mae rmse"
    error = (y - y_hat).detach().cpu()
    error = np.array(error)
    mae = np.sum(np.abs(error))
    return mae
def cutflux(wave, flux):
    returnflux = []
    for i in range(len(wave)):
        if wave[i] >= 4000.0 and wave[i] <= 7000.0:
           returnflux.append(flux[i])
        # elif wave[i] >= 5350.0 and wave[i] <= 5450.0:
        #     returnflux.append(flux[i])
        # elif wave[i] >= 6500.0 and wave[i] <= 6640.0:
        #     returnflux.append(flux[i])
        else:
            continue
    return returnflux

class MyDataset(data.Dataset):
    def __init__(self,datas,labels):
        self.Data = np.asarray(datas)  # 数据集
        self.Label = np.asarray(labels)  # 这是数据集对应的标签

    def __getitem__(self, index):
        # 把numpy转换为Tensor
        data = torch.from_numpy(self.Data[index]).reshape(1,2817)#1142#torch.from_numpy(self.Data[index]).reshape(1,1213)

        label = torch.tensor(self.Label[index])
        return data, label

    def __len__(self):
        return len(self.Data)

#def train(model,optimizer,lossfunction,epoch):
def train(model,optimizer,epoch):
    model.train()
    batch = tqdm(train_dataloader)
    total_loss = 0
    # all_outputs = []
    # all_labels = []
    for inputs, predictions in batch:
        inputs, predictions = inputs.type(torch.FloatTensor).cuda(), predictions.cuda()
        outputs= model(inputs)
        # all_outputs.extend(outputs.detach().cpu().numpy())
        # all_labels.extend(predictions.detach().cpu().numpy())
        loss=weight_mae_loss(outputs,predictions.float())
        #loss=lossfunction(outputs,predictions.float())
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # all_outputs = [output.item() for output in all_outputs]
    # all_labels = [label.item() for label in all_labels]
    # if (epoch + 1) % 20 == 0:
    #     batch.set_description('第{0}轮训练，当前损失平均值为：{1:.5f}'.format(epoch + 1, loss))
    #     print("All Outputs for Epoch {}: {}".format(epoch, all_outputs))
    #     print(len(all_outputs))
    #
    #     if all_outputs[0] > 10000:
    #         plt.figure(figsize=(25, 25))
    #         x = [20000, 80000]
    #         y = [20000, 80000]
    #         plt.plot(x, y, color='black', linewidth=2, linestyle='dotted')
    #         plt.tick_params(labelsize=25)
    #         plt.xlabel('T$_e$$_f$$_f$ (Se-ResNet)', fontsize=25)
    #         plt.ylabel('T$_e$$_f$$_f$ (Culpan2022)', fontsize=25)
    #         plt.scatter(all_outputs, all_labels, c='none', marker='o', edgecolors='teal')
    #         plt.savefig(f"/home/admin1/data1/czd/pred/draw/teff{epoch}", dpi=300)

    writer.add_scalar('Loss/Train', total_loss / len(train_dataloader), epoch)
    # 调整学习率
    cur_lr = adjust_learning_rate(optimizer, args.lr, epoch, args)

    #cur_lr = args.lr
    """
    if epoch%20==0 and epoch!=0:
        cur_lr = adjust_learning_rate2(optimizer, args.lr, epoch, args)"""
    # 保存学习率到TensorBoard
    writer.add_scalar('Learning Rate', cur_lr, epoch)
#def test(model,lossfunction, epoch):
def test(model, epoch):
    model.eval()
    total_loss = 0
    num_samples = 0
    all_outputs = []
    all_labels = []
    # 初始化 MAE
    mae = 0
    bestmae = 100000
    batch = tqdm(test_dataloader)
    with torch.no_grad():
        for inputs, labels in batch:
            inputs, labels = inputs.type(torch.FloatTensor).cuda(), labels.cuda()
            outputs = model(inputs)
            all_outputs.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            #loss=lossfunction(outputs,labels)
            loss=weight_mae_loss(outputs,labels)
            total_loss += loss.item()
            num_samples += len(inputs)

            # 使用 get_mae 函数计算 MAE
            batch_mae = get_mae(outputs, labels)#这里函数改了
            mae += batch_mae

            # 将真实值和预测值转换为 numpy 数组，用于后续计算
            true_vals = labels.cpu().numpy()
            pred_vals = outputs.cpu().numpy()

            # 输出当前测试的损失值
            #print("Test Loss (Batch): {:.5f}".format(loss.item()))
            batch.set_description('第{0}轮测试，当前损失平均值为：{1:.2f}'.format(epoch + 1, loss))
        all_outputs = [output.item() for output in all_outputs]
        all_labels = [label.item() for label in all_labels]
        mae /= num_samples
        print("mae:",mae)
        if mae < bestmae :
            torch.save(model.state_dict(), f'/home/admin1/data1/czd/pred/model2/trained_model_8-700-1e-2-teff_epoch{epoch}.pth')
            print("All Outputs for Epoch {}: {}".format(epoch, all_outputs))
            print(len(all_outputs))
            if all_outputs[0] > 10000:
                plt.figure(figsize=(10, 10))
                x = [20000, 80000]
                y = [20000, 80000]
                plt.plot(x, y, color='black', linewidth=2, linestyle='dotted')
                plt.xlim(15000, 85000)
                plt.ylim(15000, 85000)
                plt.xticks([20000, 40000, 60000, 80000])
                plt.yticks([20000, 40000, 60000, 80000])
                plt.tick_params(labelsize=25)
                plt.xlabel('T$_e$$_f$$_f$ (Se-ResNet)', fontsize=25)
                plt.ylabel('T$_e$$_f$$_f$ (Culpan2022)', fontsize=25)
                plt.scatter(all_outputs, all_labels, c='none', marker='o', edgecolors='teal')
                plt.tight_layout()
                plt.savefig(f"/home/admin1/data1/czd/pred/testdraw/teff{epoch}", dpi=300)
            bestmae = mae
        # 计算总体 MAE

        # 使用 add_scalar 方法将 MAE 写入 TensorBoard 日志
        writer.add_scalar('MAE', mae, epoch)

        # 保存总体损失值到 TensorBoard
        writer.add_scalar('Loss/Test', total_loss / len(test_dataloader), epoch)

    # 关闭 SummaryWriter
    writer.close()

def adjust_learning_rate2(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr /10
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    return cur_lr




def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    return cur_lr

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train_and_test')
    parser.add_argument('--batch_size', type=int, default=8 ,help='batch size')
    parser.add_argument('--epochs', type=int, default=700, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='epochs')
    args = parser.parse_args()

    # 开始计时#######################################################################
    # start = time.perf_counter
    # #
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    obsid=[]
#################################数据集划分####################
    hdu1 = fits.open('/home/admin1/data1/czd/pred/tefftrain.fits')
    for i in range(len(hdu1[1].data)) :
        temp = []

        train_x.append(cutflux(hdu1[1].data[i]['wavelength'],hdu1[1].data[i]['flux']))
        temp.append(hdu1[1].data[i]['teff'])
        train_y.append(temp)

    hdu2 = fits.open('/home/admin1/data1/czd/pred/tefftest.fits')
    for i in range(len(hdu2[1].data)) :
        temp = []

        test_x.append(cutflux(hdu2[1].data[i]['wavelength'],hdu2[1].data[i]['flux']))
        temp.append(hdu2[1].data[i]['teff'])
        test_y.append(temp)

   # train_x,test_x,train_y,test_y =train_test_split(all_x, all_y, test_size=0.3,random_state=15)
    train_dataset = MyDataset(train_x,train_y)

    test_dataset = MyDataset(test_x, test_y)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=0)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=0)

    model = Model.Get_se_resnet_19_1d(num_classes=1).cuda()#DenseNEt.densenet_BC_17().cuda()#Model.Get_se_resnet_19_1d(num_classes=1).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 初始化 TensorBoard 的 SummaryWriter
    writer = SummaryWriter("/home/admin1/data1/czd/pred/logs2/8-700-1e-2-teff/")

    ###########################################################
    #lossfunction=nn.MSELoss(reduction='mean')#nn.L1Loss()#nn.SmoothL1Loss()#nn.L1Loss()#nn.MSELoss(reduction='mean')
    #lossfunction = HuberLoss(delta=0.1)
    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer,args.lr,epoch,args)

        #train(model, optimizer,lossfunction,epoch)

        train(model, optimizer,epoch)

        #test(model,lossfunction,epoch)
        test(model,epoch)
    # 关闭 SummaryWriter
    writer.close()
    # 保存训练好的模型
    #torch.save(model.state_dict(), '/home/admin1/data1/czd/pred/model2/trained_model8-1000-1e-4-teff.pth')






