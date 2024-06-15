import csv
from astropy.io import fits
import os
import numpy as np
from astropy.table import QTable, Table, Column
from scipy import interpolate
import copy
import pandas as pd
import random
import string
from tqdm import tqdm

def changeRV(wavelength):  # 共同函数
    wave0 = []
    for i in wavelength:
        wave0.append(i)
    return wave0


def removeExpval(data):
    dd = copy.deepcopy(data)
    # 深复制deepcopy，就是从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响
    # 而浅复制copy不是，它占用的内存和原变量是一样的
    np.sort(dd)  # 数组排序
    max1 = dd[-1]  # 就是读最后一个
    max2 = dd[-2]
    if max1 >= 1.5 * max2:
        return max1
    else:
        return 0


# 寻找相同obsid所在位置
def FindSame(hdu, obsid):
    for i in range(len(hdu)):
        if (hdu['obsid'][i] == obsid):
            return i


# 对非871
obsid = []
wavelengthh = []
fluxx = []
label = []
n = 0
rootdir = "/home/admin1/data1/czd/czd_lbw/180spec/"    # 根目录
list = os.listdir(rootdir)  # 返回一个列表，其中包含由rootdir指定的目录中的条目的名称
print(len(list))
for j in tqdm(range(0, len(list))):  # len(list)
    path = os.path.join(rootdir, list[j])
    if os.path.isfile(path):
        try:
            hdu = fits.open(path)
        except Exception:
            print(path)
            continue
        hdr = hdu[0].header  # 看No.0的头文件（类似于目录）
        
        if j >= 0:
            try:
                orgwavelength = changeRV(hdu[1].data[0][2])  # 读数据，都加到数组orgwavelength里面
                orgflux = hdu[1].data[0][0]
            except:
                orgwavelength = changeRV(hdu[0].data[2]) 
                orgflux = hdu[0].data[0]
            
            aflux = []
            awave = []
            maxvalue = removeExpval(orgflux)
            if maxvalue != 0:
                # print('test4')
                for f in range(len(orgflux)):
                    if orgflux[f] < maxvalue:
                        aflux.append(orgflux[f])
                        awave.append(orgwavelength[f])
            else:
              
                awave = orgwavelength
                aflux = orgflux
                flux = (aflux - np.min(aflux)) / (np.max(aflux) - np.min(aflux))
                wavenew = np.linspace(4000, 8000, 3909)  # 用等差数列创造一组数
               
                f = interpolate.interp1d(awave, flux, kind="slinear")
                
                fluxnew = f(wavenew)

            wavelengthh.append(wavenew)
            fluxx.append(fluxnew)  # 为什么不用归一化光谱
            
            obsid.append(hdr['OBSID'])
            
            label.append(1)
                
t = Table([obsid, wavelengthh, fluxx],
          names=('obsid', 'wavelength', 'flux'))
print(t.info)
t.write('/home/admin1/data1/czd/czd_lbw/180.fits', format='fits', overwrite=True)