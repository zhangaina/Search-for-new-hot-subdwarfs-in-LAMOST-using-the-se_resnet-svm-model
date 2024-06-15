import argparse
import random
import os
from astropy.table import Table
from astropy.io import fits
from scipy import interpolate
import numpy as np
from astropy.table import QTable, Table, Column
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from matplotlib.patches import ConnectionPatch
import copy
import torch.nn.functional as F  # 激励函数都在这

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
from sklearn.metrics import f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
import Model
# import DenseNEt
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import math
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
#训练se_resnet网络

class MyDataset(data.Dataset):
    def __init__(self, datas, labels):
        self.Data = np.asarray(datas)  # 数据集
        self.Label = np.asarray(labels)  # 这是数据集对应的标签

    def __getitem__(self, index):
        # 把numpy转换为Tensor

        data = torch.from_numpy(self.Data[index]).reshape(1,
                                                          3909)  # 1142 #torch.from_numpy(self.Data[index]).reshape(1,1213)

        label = torch.tensor(self.Label[index])
        return data, label

    def __len__(self):
        return len(self.Data)


def Calculate_acc_ppv_sen_spec(matrix, class_num, class_names):
    results_matrix = np.zeros([class_num, 4])
    # diagonal负责统计对角线元素的和
    diagonal = 0
    for i in range(class_num):
        tp = matrix[i][i]
        diagonal += tp
        fn = np.sum(matrix, axis=1)[i] - tp
        fp = np.sum(matrix, axis=0)[i] - tp
        tn = np.sum(matrix) - tp - fp - fn
        acc = (tp + tn) / (tp + tn + fp + fn)
        ppv = tp / (tp + fp)
        sen = tp / (tp + fn)
        spec = tn / (tn + fp)
        results_matrix[i][0] = acc * 100
        results_matrix[i][1] = ppv * 100
        results_matrix[i][2] = sen * 100
        results_matrix[i][3] = spec * 100

    average = [0 for i in range(4)]

    for i in range(class_num):
        print('{0}：acc:{1:.2f}%,ppv:{2:.2f}%,sen:{3:.2f}%,spec:{4:.2f}%'.format(class_names[i], results_matrix[i][0],
                                                                                results_matrix[i][1],
                                                                                results_matrix[i][2],
                                                                                results_matrix[i][3]))
        average[0] += results_matrix[i][0]
        average[1] += results_matrix[i][1]
        average[2] += results_matrix[i][2]
        average[3] += results_matrix[i][3]

    print('四项评价指标求均值,acc:{0:.2f}%,ppv:{1:.2f}%,sen:{2:.2f}%,spec:{3:.2f}%'.format(average[0] / class_num,
                                                                                           average[1] / class_num,
                                                                                           average[2] / class_num,
                                                                                           average[3] / class_num))


# 计算模型的总体acc,ppv,sen,spec
def Calculate_total_acc_ppv_sen_spec(matrix):
    tn = matrix[0][0]
    fp = np.sum(matrix, axis=1)[0] - tn
    fn = np.sum(matrix, axis=0)[0] - tn
    tp = np.sum(matrix) - tn - fp - fn
    acc = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp)
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    print('tp:{0},tn:{1},fp:{2},fn:{3}'.format(tp, tn, fp, fn))
    print('total：acc:{0:.2f}%,ppv:{1:.2f}%,sen:{2:.2f}%,spec:{3:.2f}%'.format(acc * 100, ppv * 100, sen * 100,
                                                                              spec * 100))


def train(model, optimizer, lossfunction, epoch):
    model.train()
    batch = tqdm(train_dataloader)
    for inputs, predictions in batch:
        inputs, predictions = inputs.type(torch.FloatTensor).cuda(), predictions.type(torch.LongTensor).cuda()

        outputs, feature1 = model(inputs)

        # loss=lossfunction(outputs.type(torch.FloatTensor),predictions.type(torch.FloatTensor))
        loss = lossfunction(outputs, predictions)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch.set_description('第{0}轮训练，当前损失平均值为：{1:.5f}'.format(epoch + 1, loss))


def test(model, lossfunction):
    model.eval()
    total_loss = 0
    trueLog = []
    trueTeff = []
    trueHeh = []
    predictionLog = []
    predictionTeff = []
    predictionHeh = []
    num = 0
    # x=[4.6,6.8]
    # y=[4.6,6.8]
    # # x=[1800,80000]
    # # y=[1800,80000]
    feature = []
    y_true = []

    y_pred = []
    y_pred1 = []
    n = 0
    batch = tqdm(test_dataloader)
    with torch.no_grad():
        for inputs, predictions in batch:
            inputs, predictions = inputs.type(torch.FloatTensor).cuda(), predictions.type(torch.LongTensor).cuda()
            outputs, temp3 = model(inputs)
            loss = lossfunction(outputs, predictions)

            prediction = torch.max(F.softmax(outputs, dim=1), 1)[1]

            # print(prediction)
            # total_lo
            # ss+=loss.item()
            # prediction = F.softmax(outputs)

            # print(prediction)

            # # print(outputs)
            for i in range(len(prediction)):
                y_pred.append(prediction[i].item())
                y_true.append(predictions[i].item())

            ####### extract abstarct feature################
            # for x,y in zip(temp3.data.cpu().numpy(),predictions.data.cpu().numpy()):
            #     all_data.append(x)
            #     all_label.append(y)
            # t = Table([obsid, all_data, all_label], names=('obsid', 'flux', 'label'))

            # t.write(r'BMIX.fits', format='fits')

    # #
    #
    print('测试集acc为：{0:.3f}%'.format(accuracy_score(y_true, y_pred) * 100))
    print('测试集f1_score为：{0:.3f}%'.format(f1_score(y_true, y_pred) * 100))
    print('测试集recall_score为：{0:.3f}%'.format(recall_score(y_true, y_pred) * 100))
    print('测试集precision_score为：{0:.3f}%'.format(precision_score(y_true, y_pred) * 100))

    # cm = confusion_matrix(y_true, y_pred)


##################################################################
# def test2(model,lossfunction,obsid):
#     model.eval()
#     total_loss=0
#     trueLog=[]
#     trueTeff=[]
#     trueHeh=[]
#     predictionLog=[]
#     predictionTeff=[]
#     predictionHeh=[]
#     num=0
#     # x=[4.6,6.8]
#     # y=[4.6,6.8]
#     # # x=[1800,80000]
#     # # y=[1800,80000]
#     feature = []
#     y_true = []
#
#     y_pred = []
#     y_pred1 = []
#     n=0
#     batch = tqdm(test_dataloader)
#     with torch.no_grad():
#         for inputs, predictions in batch:
#             inputs, predictions = inputs.type(torch.FloatTensor).cuda(), predictions.type(torch.LongTensor).cuda()
#             outputs,temp3 = model(inputs)
#             loss=lossfunction(outputs,predictions)
#
#             prediction  = torch.max(F.softmax(outputs), 1)[1]
#
#
#             # print(prediction)
#             # total_lo
#             # ss+=loss.item()
#             # prediction = F.softmax(outputs)
#
#             # print(prediction)
#
#             # # print(outputs)
#             for i in range(len(prediction)):
#                 y_pred.append(prediction[i].item())
#                 y_true.append(predictions[i].item())
#
#
#             ####### extract abstarct feature################
#             for x,y in zip(temp3.data.cpu().numpy(),predictions.data.cpu().numpy()):
#                 all_data.append(x)
#                 all_label.append(y)
#             t = Table([obsid, all_data, all_label], names=('obsid', 'flux', 'label'))
#
#             t.write(r'BMIX.fits', format='fits')
################################################

def cutflux(wave, flux):  ########using for get artificial features
    returnflux = []
    for i in range(len(wave)):
        if wave[i] >= 4000.0 and wave[i] <= 5000.0:
            returnflux.append(flux[i])
        elif wave[i] >= 5350.0 and wave[i] <= 5450.0:
            returnflux.append(flux[i])
        elif wave[i] >= 6500.0 and wave[i] <= 6640.0:
            returnflux.append(flux[i])
        else:
            continue
        print(wave[i], len(returnflux))
    return returnflux


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_and_test')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='epochs')
    args = parser.parse_args()

    # 开始计时
    # start = time.perf_counter
    # #
    # filepath1 = '/home/admin1/MJQ/output/2905.fits'
    filepath1 = '/home/admin1/data1/czd/hot.fits'
    filepath2 = "/home/admin1/data1/czd/negatives_test_noWD.fits"
    filepath3 = "/home/admin1/data1/czd/negatives_train_noWD.fits"

    """
    filepath2 = "/home/admin1/data/mjq/negatives_test_noWD.fits"
    filepath3 = "/home/admin1/data/mjq/negatives_train_noWD.fits"
    """

    hdu = fits.open(filepath1)

    all_x = []
    all_y = []

    for i in range(len(hdu[1].data)):
        all_x.append(hdu[1].data[i]['flux'])
        all_y.append(np.array(1))

    train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=0.3)

    hdu = fits.open(filepath2)

    for i in range(len(hdu[1].data)):
        test_x.append(hdu[1].data[i]['flux'])
        test_y.append(np.array(0))

    hdu = fits.open(filepath3)

    for i in range(len(hdu[1].data)):
        train_x.append(hdu[1].data[i]['flux'])
        train_y.append(np.array(0))

    test_x, test_y = shuffle(test_x, test_y)
    train_x, train_y = shuffle(train_x, train_y)

    train_dataset = MyDataset(train_x, train_y)

    test_dataset = MyDataset(test_x, test_y)
    #
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    #
    model = Model.Get_se_resnet_19_1d(
        num_classes=2).cuda()  # DenseNEt.densenet_BC_17().cuda()#Model.Get_se_resnet_19_1d(num_classes=1).cuda()
    #
    #
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lossfunction = nn.CrossEntropyLoss().cuda()  # nn.MSELoss(reduction='mean')#nn.L1Loss()#nn.SmoothL1Loss()#nn.L1Loss()#nn.MSELoss(reduction='mean')

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, args.lr, epoch, args)

        train(model, optimizer, lossfunction, epoch)

        test(model, lossfunction)

    torch.save(model, '/home/admin1/data1/czd/two-class-model/2905_noWD.pkl')


"""
    model = torch.load('bhbBmixfeature.pkl')
    #############start to train svm for classification###############
    from sklearn import svm

    bhb = 'mixbhb.fits'

    # b = 'BMIX.fits'

    # houxuanti = 'mixhouxuanti.fits'

    newHot = 'mixbhbnewHot.fits'

    pathfile = 'mixbhb.fits'
    hdu = fits.open(newHot)
    all_data = []
    all_label = []
    tdata = []
    fdata = []
    filepath1 = 'D:\\newHot.fits'
    wavelength = fits.open(filepath1)
    waveleng = wavelength[1].data[1][2]
    for i in range(len(hdu[1].data)):
        temp = hdu[1].data[i][1][0][3909:]
        print(len(cutflux(waveleng, hdu[1].data[i][1][0][0:3909])))
        temp = np.append(cutflux(waveleng, hdu[1].data[i][1][0][0:3909]), temp)
        all_data.append(temp)
        all_label.append(0)
    length = len(all_data)
    hdu = fits.open(bhb)
    for i in range(length):
        temp = hdu[1].data[i][1][0][3909:]
        temp = np.append(cutflux(waveleng, hdu[1].data[i][1][0][0:3909]), temp)
        all_data.append(temp)
        all_label.append(1)
    hdu = fits.open(bhb)
    for i in range(length):
        temp = hdu[1].data[i][1][0][3909:]
        temp = np.append(cutflux(waveleng, hdu[1].data[i][1][0][0:3909]), temp)
        all_data.append(temp)
        all_label.append(2)
    train_x, test_x, train_y, test_y = train_test_split(all_data, all_label, test_size=0.3)

    classifier = svm.SVC(C=10000, kernel='rbf', gamma=0.000001, decision_function_shape='ovr', probability=1)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict_proba(test_x)
    cm = confusion_matrix(test_y, y_pred)
    y_pred = classifier.predict(test_x)
    
"""