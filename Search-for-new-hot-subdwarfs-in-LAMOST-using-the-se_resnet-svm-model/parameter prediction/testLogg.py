import argparse
import numpy as np
import time
from astropy.io import fits
import matplotlib.pyplot as plt

import os
from torch.utils import data
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import Model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import math
from sklearn.metrics import confusion_matrix

class MyDataset(data.Dataset):
    def __init__(self,datas,labels):
        self.Data = np.asarray(datas)  # 数据集
        self.Label = np.asarray(labels)  # 这是数据集对应的标签

    def __getitem__(self, index):
        # 把numpy转换为Tensor
        data = torch.from_numpy(self.Data[index]).reshape(1,3909)
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

#计算模型的总体acc,ppv,sen,spec
def Calculate_total_acc_ppv_sen_spec(matrix):
    tn=matrix[0][0]
    fp=np.sum(matrix,axis=1)[0]-tn
    fn=np.sum(matrix,axis=0)[0]-tn
    tp=np.sum(matrix)-tn-fp-fn
    acc = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp)
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    print('tp:{0},tn:{1},fp:{2},fn:{3}'.format(tp,tn,fp,fn))
    print('total：acc:{0:.2f}%,ppv:{1:.2f}%,sen:{2:.2f}%,spec:{3:.2f}%'.format(acc*100,ppv*100,sen*100,spec*100))

def train(model,optimizer,lossfunction,epoch):
    model.train()
    batch = tqdm(train_dataloader)
    for inputs, predictions in batch:
        inputs, predictions = inputs.type(torch.FloatTensor).cuda(), predictions.unsqueeze(dim=-1).cuda()
        outputs = model(inputs)

        loss=lossfunction(outputs.type(torch.FloatTensor),predictions.type(torch.FloatTensor))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch.set_description('第{0}轮训练，当前损失平均值为：{1:.5f}'.format(epoch + 1, loss))

def test(model,lossfunction):
    model.eval()
    total_loss=0
    num=0
    trueLog=[]
    predictionLog=[]
    batch = tqdm(test_dataloader)
    with torch.no_grad():
        for inputs, predictions in batch:
            inputs, predictions = inputs.type(torch.FloatTensor).cuda(), predictions.unsqueeze(dim=-1).cuda()
            outputs = model(inputs)
            for i in predictions:
                predictionLog.append(i.item())
            for i in outputs:
                trueLog.append(i.item())
            loss=lossfunction(outputs.type(torch.FloatTensor),predictions.type(torch.FloatTensor))
            total_loss+=loss.item()
            num+=1
            print(num)
    print('测试集平均差值为:{0:.3f}'.format(total_loss/num))
    return predictionLog , trueLog

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
    parser.add_argument('--batch_size', type=int, default=32 ,help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-1, help='epochs')
    args = parser.parse_args()

    # 开始计时
    start = time.perf_counter()

    mydata = fits.open('D:\\project\\hot\\hotsubwardf1.fits')
    all_x=[]
    all_y=[]
    for i in range(len(mydata[1].data)):
        all_x.append(mydata[1].data[i][6])
        all_y.append(mydata[1].data[i][4])
    all_x = np.asarray(all_x)
    all_y = np.asarray(all_y)

    #all_x = o
    # all_x = np.load(os.path.join('./', 'data_save', 'AllData.npy'))
    # all_y = np.load(os.path.join('./', 'data_save', 'AllLabel.npy'))
    # print(all_x)

    num=0

    #训练集按照9：1划分训练集与验证集
    train_x,test_x, train_y,test_y = train_test_split(all_x, all_y, test_size=0.2,random_state=52)

    train_dataset = MyDataset(train_x, train_y)
    test_dataset = MyDataset(test_x, test_y)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,num_workers=0)

    model = torch.load('DenseNetSVMSr6666.pkl').cuda()

    lossfunction=nn.MSELoss(reduction='mean')

    prediction , outputs = test(model, lossfunction)
    # x=[4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6,6.8,7.0]
    # y=[4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6,6.8,7.0]
    x=[-4.5 , -4.0 , -3.5 , -3.0 , -2.5 , -2.0 , -1.5 , -1.0 , -0.5 , 0 , 0.5 , 1 , 1.5 , 2.0 ,2.5 , 3.0]
    y=[-4.5 , -4.0 , -3.5 , -3.0 , -2.5 , -2.0 , -1.5 , -1.0 , -0.5 , 0 , 0.5 , 1 , 1.5 , 2.0 ,2.5 , 3.0]
    plt.plot(x,y)
    plt.xlabel('prediction')
    plt.ylabel('true')
    plt.scatter(prediction,outputs)
    # plt.savefig('He/H.png')
    plt.show()



