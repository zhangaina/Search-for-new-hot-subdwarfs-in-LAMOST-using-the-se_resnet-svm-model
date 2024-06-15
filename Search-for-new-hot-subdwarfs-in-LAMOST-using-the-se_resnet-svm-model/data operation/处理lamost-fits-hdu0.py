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


# def changeRV(wavelength):  # 共同函数
#     wave0 = []
#     for i in wavelength:
#         wave0.append(i)
#     return wave0
#
#
# def cutflux(wave, flux):  # 波段截取
#     returnflux = []
#     for i in range(len(wave)):
#         if wave[i] >= 4000.0 and wave[i] <= 8000.0:
#             returnflux.append(flux[i])
#         # if wave[i] >= 4000.0 and wave[i] <= 5000.0:
#         #     returnflux.append(flux[i])
#         # elif wave[i] >= 5350.0 and wave[i] <= 5450.0:
#         #     returnflux.append(flux[i])
#         # elif wave[i] >= 6500.0 and wave[i] <= 6640.0:
#         #     returnflux.append(flux[i])
#         else:
#             continue
#     return returnflux

#
# def removeExpval(data):
#     dd = copy.deepcopy(data)
#     np.sort(dd)
#     max1 = dd[-1]
#     max2 = dd[-2]
#     if max1 >= 1.5 * max2:
#         return max1
#     else:
#         return 0
#
#
# def FindSameloc(hdu, source_id):
#     for i in range(len(hdu)):
#         if (hdu['dr3_source_id'][i] == source_id):
#             return i
#
#
# def FindSameRaDec(hdu, ra, dec):
#     for i in range(len(hdu)):
#         if (abs(hdu['ra'][i] - ra) <= 0.000001) and (abs(hdu['dec'][i] - dec) <= 0.000001):
#             return i
# #从lamost下载的fits文件会有多个
#
# text = "D:\\candiate\\mixfeature\\txt\\" #想要依次读取一个文件夹下的所有fits文件
# dirs = os.listdir(text)
# fluxx=[]
# namee=[]
# maxLen = 0
# for i in range(len(dirs)):  #len(dirs)
#     # print(i)
#     path = os.path.join(text, dirs[i])
#     print(path)
#     fitsname = dirs[i].split('.fits')[0]
#     #把文件类型删掉只留下文件名 比如文件名为a.fits,split('.fits')就是把它分成a和.fits,是数组形式['a','.fits'],split('.fits')[0]就是a
#     n=1
#     wave = []
#     flux = []
#     #print(directory)                # 打印txt文件名
#     with open(path, "r") as f:  # 打开txt文件
#         for line in f.readlines():
#             if n == 1 :
#                 n = n + 1
#             else:
#                 data = []
#                 temp = line.split() #以所有的空字符来分割，分割完之后存在数组temp中
#                 data.append(temp)
#                 wave.append(float(data[0][0]))
#                 flux.append(float(data[0][1]))
#
#             # for i in range(1,len(data)):
#
#
#     newflux  = cutflux(wave,flux)
#
#     if len(newflux) != 1142:
#         continue
#
#     fitsname = fitsname + ".fits" + ".gz" #.gz压缩
#     #
#     fluxx.append(newflux)
#     namee.append(fitsname)
#
# t = Table(data = [namee,fluxx], names=['namee','fluxx'])
# 变成csv
# t.write(r'mixfeature.fits', format='fits')
# pd.DataFrame(point).to_csv('confirmdata.csv')
# 或者变成fits
# t = Table([namee,fluxx], names=('obsid','flux','label'))
# t.write(r'fenleiunknown.fits', format='fits')

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
# teff = []
# logg = []
# fe_h = []

wavelengthh = []
fluxx = []
label = []
# desig = []
n = 0
# idd871=pd.read_csv('D:\Vdr8test\obsid871.csv')
# idd=idd871['obsID']
rootdir = "/home/admin1/data1/czd/czd_lbw/159spec/"    # 根目录
list = os.listdir(rootdir)  # 返回一个列表，其中包含由rootdir指定的目录中的条目的名称
# weizhi='D:\\nork\\20130.csv'
# nork = 'D:/project/trainBa.fits'
# nork = fits.open(nork)
# Ba = []
# Sr = []
# BaSr = []
# hdu1 = pd.read_csv('D:\\Users\\songk\\Desktop\\dch.csv')
# hdu=fits.open('D:\\Users\\songk\\Desktop\\a\\spec-55875-GAC_072N28_B1_sp01-139.fits')
# print(hdu[1].data[0][0])
# hdu2=fits.open('D:\\Users\\songk\\Desktop\\a\\spec-55875-GAC_072N28_B1_sp01-245.fits')
# print(hdu2[1].data[0][0])
print(len(list))
for j in tqdm(range(0, len(list))):  # len(list)
    # os.path.join()函数用于路径拼接文件路径，可以传入多个路径，print(os.path.join('path','abc','yyy'))的结果是path\abc\yyy
    path = os.path.join(rootdir, list[j])
    # os.path.isfile（）判断是否是文件
    # print('test1')
    if os.path.isfile(path):
        # print('test2')
        try:
            hdu = fits.open(path)
        except Exception:
            print(path)
            continue
        # print(path)
        #         #fits，它是天文学界常用的数据格式，它专门为在不同平台之间交换数据而设计
        hdr = hdu[0].header  # 看No.0的头文件（类似于目录）
        # loc = FindSame(hdu1, hdr['OBSID'])
        # print(loc)
        # print(hdu[1].data[0][2])
        #                hduall = pd.read_csv(weizhi,sep='|')
        #                #read_csv是按照某分隔符读取csv文件，默认分隔符是逗号
        #                for i in range(0,100):
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
                # print('test5')
                awave = orgwavelength
                aflux = orgflux
                flux = (aflux - np.min(aflux)) / (np.max(aflux) - np.min(aflux))
                wavenew = np.linspace(4000, 8000, 3909)  # 用等差数列创造一组数
                # linspace用于在线性空间中以均匀步长生成数字序列，通常用numpy.arange()生成序列，但是当我们使用浮点参数时，可能会导致精度损失
                # 格式：array = numpy.linspace(start, end, num=num_points)将在start和end之间生成一个统一的序列，共有num_points个元素
                f = interpolate.interp1d(awave, flux, kind="slinear")
                # scipy.interpolate是插值模块interp1d一维插值，返回一个函数
                # interpolate.interp1d(x, y, kind=……）x和y是用来逼近函数f: y = f(x)的值的数组
                fluxnew = f(wavenew)

            wavelengthh.append(wavenew)
            # print(awave)
            fluxx.append(fluxnew)  # 为什么不用归一化光谱
            # print(aflux)
            # teff.append(hdu1['teff'][loc])
            # # print(hdu1['teff'][loc])
            # logg.append(hdu1['logg'][loc])
            # fe_h.append(hdu1['feh'][loc])
            obsid.append(hdr['OBSID'])
            # desig.append(hdr['DESIG'])
            label.append(1)
                # n += 1
                # break
# t = Table([obsid, teff, logg, fe_h, label, wavelengthh, fluxx],
#           names=('obsid', 'teff', 'logg', 'fe_h', 'label', 'wavelength', 'flux'))
# print(len(wavelengthh[1]))
# for i in range(len(wavelengthh)):
#     print(len(wavelengthh[i]))
# wavelengthh = np.array(wavelengthh)
# fluxx = np.array((fluxx))
# print(wavelengthh.shape)
# t = Table([obsid, teff, logg, fe_h, label, wavelengthh, fluxx],
#           names=('obsid', 'teff', 'logg', 'fe_h', 'label', 'wavelength', 'flux'))
t = Table([obsid, wavelengthh, fluxx],
          names=('obsid', 'wavelength', 'flux'))
print(t.info)
t.write('/home/admin1/data1/czd/czd_lbw/159.fits', format='fits', overwrite=True)