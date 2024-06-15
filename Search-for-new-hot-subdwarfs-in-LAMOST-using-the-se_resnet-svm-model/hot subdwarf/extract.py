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
import torch.nn.functional as F   # 激励函数都在这

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
# import DenseNEt
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import math
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
#使用训练好的se_resnet网络提取特征然后与特征带拼接





#正样本负样本进行特征提取
class MyDataset(data.Dataset):
    def __init__(self,datas,labels):
        self.Data = np.asarray(datas)  # 数据集
        self.Label = np.asarray(labels)  # 这是数据集对应的标签

    def __getitem__(self, index):
        # 把numpy转换为Tensor

        data =  torch.from_numpy(self.Data[index]).reshape(1,3909) #1142 #torch.from_numpy(self.Data[index]).reshape(1,1213)

        label = torch.tensor(self.Label[index])
        return data, label

    def __len__(self):
        return len(self.Data)
    
def cutflux(wave, flux):########using for get artificial features
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
        # print(wave[i], len(returnflux))
    return returnflux

def extract(filepath, model, outputpath):

    flux = []
    label = []
    obsid = []
    hdu = fits.open(filepath)

    all_data = []
    all_label = []

    waveleng = hdu[1].data[1]['wavelength']


    for i in range(len(hdu[1].data)):
        #waveleng = hdu[1].data[i]['wavelength']
        flux.append(hdu[1].data[i]['flux'])
        obsid.append(hdu[1].data[i]['obsid'])
        label.append(np.array(0))

        #temp = hdu[1].data[i]['flux'][3909:]
        #截取那三段，然后拼起来
        #temp = np.append(cutflux(waveleng, hdu[1].data[i]['flux'][0:3909]), temp)
        temp = cutflux(waveleng, hdu[1].data[i]['flux'][0:3909])
        all_data.append(temp)
        all_label.append(1)
        #print(temp)

    

    dataset = MyDataset(flux, label)


    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False,num_workers=0)
    batch = tqdm(dataloader)

    features = []
    #把4000-8000扔到模型里面
    with torch.no_grad():
        for i, (inputs, predictions) in enumerate(batch):
            inputs = inputs.type(torch.FloatTensor).cuda() 
            outputs, feature = model(inputs)

            for x in feature.data.cpu().numpy():
                features.append(x)

        # for item in features[0]:
        #     print(item, end=' ')
        # print()  # 在列表结束后打印换行

        for i in range(len(features)):
            min_val = np.min(features[i])
            max_val = np.max(features[i])
            # print("max:", max_val)
            # print("min:", min_val)
            if max_val > min_val:
                features[i] = (features[i] - min_val) / (max_val - min_val)
            else:
                features[i] = 0

        # for item in features[0]:
        #     print(item, end=' ')
        # print()  # 在列表结束后打印换行

        all_data = np.hstack((all_data, features))  
        # exchange

        t = Table([obsid, all_data, all_label], names=('obsid', 'flux', 'wavelength'))
        t.write(outputpath, format='fits', overwrite=True)
    
    

if __name__=='__main__':
    filepath1 = '/home/admin1/data1/czd/hot.fits'
    filepath2 = "/home/admin1/data1/czd/negatives_test.fits"
    filepath3 = "/home/admin1/data1/czd/negatives_train.fits"



    model = torch.load('/home/admin1/data1/czd/two-class-model/2905_WD64-200-1e-3.pkl')
    """
    outputpath1 = r'/home/admin1/data1/czd/two-class-model/extract/exhots_noWD1-i.fits'
    outputpath2 = r'/home/admin1/data1/czd/two-class-model/extract/exnegatives_test_noWD1-i.fits'
    outputpath3 = r'/home/admin1/data1/czd/two-class-model/extract/exnegatives_train_noWD1-i.fits'
    """
    outputpath1 = r'/home/admin1/data1/czd/two-class-model/extract/exhots_WD.fits'
    outputpath2 = r'/home/admin1/data1/czd/two-class-model/extract/exnegatives_test_WD.fits'
    outputpath3 = r'/home/admin1/data1/czd/two-class-model/extract/exnegatives_train_WD.fits'

    # model = torch.load('bhbBmixfeature.pkl')


    # outputpath1 = r'/home/admin1/data/mjq/HSD/extract/exnewhots.fits'
    # outputpath2 = r'/home/admin1/data/mjq/HSD/extract/exnegatives_test.fits'
    # outputpath3 = r'/home/admin1/data/mjq/HSD/extract/exnegatives_train.fits'

    extract(filepath1, model, outputpath1)
    extract(filepath2, model, outputpath2)
    extract(filepath3, model, outputpath3)
    """
    dir = r"'/home/admin1/data/LBW/czd/hsd.fits'"
    outputpath3163 = r'/home/admin1/data/mjq/HSD/extract/exhots_noWD.fits'
    extract(dir, model, outputpath3163)
    """