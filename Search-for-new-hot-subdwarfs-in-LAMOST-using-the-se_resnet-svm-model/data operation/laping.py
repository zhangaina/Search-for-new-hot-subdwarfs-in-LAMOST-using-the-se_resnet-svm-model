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
import math
from scipy.interpolate import UnivariateSpline


def changeRV(wavelength):  # 共同函数
    wave0 = []
    for i in wavelength:
        wave0.append(i)
    return wave0


def removeExpval(data):  # 这是在干嘛？？？？？
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



#--------------------------------------------------------------------------------------
def uncertainty(spec_dir, spec_name, wave_split):
    # print(spec_dir)
    # print(spec_name)
    f = fits.open(spec_dir + spec_name, memmap=False)
    flux_air = f[1].data[0][0]
    wave_vac = f[1].data[0][2]
    inverse = f[1].data[0][1]
    for mm in range(len(flux_air)):
        if flux_air[mm] <= 0.0 or inverse[mm] <= 0.0:
            inverse[mm] = 1.0
    uncertainty_origin = np.sqrt(1.0 / inverse)
    wave_air = wave_vac / (1.0 + 2.735182E-4 + 131.4182 / wave_vac ** 2 + 2.76249E+8 / wave_vac ** 4)
    f.close()

    wave = [[] for x in range(len(wave_split))]
    # print(len(wave_split))
    flux = [[] for x in range(len(wave_split))]
    uncertainty = [[] for x in range(len(wave_split))]
    for j in range(len(wave_split)):
        for k in range(len(wave_air)):
            if j == 0:
                if wave_air[k] <= wave_split[j]:
                    wave[j].append(wave_air[k])
                    flux[j].append(flux_air[k])
                    uncertainty[j].append(uncertainty_origin[k])
            else:
                if wave_air[k] >= wave_split[j - 1] and wave_air[k] <= wave_split[j]:
                    wave[j].append(wave_air[k])
                    flux[j].append(flux_air[k])
                    uncertainty[j].append(uncertainty_origin[k])
    # print(uncertainty)
    return wave,flux,uncertainty


# --------------------------------------------------------------------------------------------
def normalize(wave, flux, uncertainty, wave_split):
    r0 = 0.5
    flux_standard = max([max(x) for x in flux if x != []])
    # print("test3")
    # print(flux)

    wave_total = []
    flux_total = []
    flux_smooth_total = []
    flux_nor_total = []
    flux_nor_err_total = []

    for nn in range(len(wave)):
        # print(wave[nn])
        if wave[nn] == []: continue
        if max(wave[nn]) <= wave_split[0]:
            wave_grid = 15
        elif max(wave[nn]) <= wave_split[1]:
            wave_grid = 120
        # elif max(wave[nn])<=wave_split[2]:wave_grid=100
        else:
            wave_grid = 40
        # print(wave_grid)

        win_num = int((max(wave[nn]) - min(wave[nn])) / wave_grid)
        # print(win_num)

        uncertainty[nn] = np.array(uncertainty[nn]) / np.array(flux[nn])
        flux[nn] = np.array(flux[nn]) / flux_standard

        n_point = int(math.ceil(len(flux[nn]) / win_num))
        # print(n_point)
        n_start = 0
        win_wave = []
        win_flux = []
        # win_flux_err=[]

        for i in range(win_num):
            w = []
            f = []
            # err=[]
            for j in range(n_start, len(flux[nn])):
                if len(w) >= n_point: continue
                w.append(wave[nn][j])
                f.append(flux[nn][j])
                # err.append(uncertainty[nn][j])

            if len(f) != 0:
                f_avg = sum(f) / len(f)
                for k in range(len(f)):
                    if w[k] <= 4200.0:
                        flux_width = 0.06
                    elif w[k] < 5800.0:
                        flux_width = 0.0035
                    elif w[k] < 8000.0:
                        flux_width = 0.0035
                    else:
                        flux_width = 0.005
                    if abs(f[k] - f_avg) <= flux_width and f[k] > 0.0:
                        win_wave.append(w[k])
                        win_flux.append(f[k])
                        # win_flux_err.append(err[k])
            n_start += n_point

        if len(win_flux) <= 3 or win_flux == []: continue
        t_del = int(len(win_wave) / 30 + 2)
        for x in range(t_del):
            win_wave_new = copy.deepcopy(win_wave)
            win_flux_new = copy.deepcopy(win_flux)
            # win_flux_err_new=copy.deepcopy(win_flux_err)
            for i in range(1, len(win_wave_new) - 1):
                if win_flux_new[i] < win_flux_new[i - 1] and win_flux_new[i] < win_flux_new[i + 1]:
                    win_wave.remove(win_wave_new[i])
                    win_flux.remove(win_flux_new[i])
                    # win_flux_err.remove(win_flux_err_new[i])

        win_wave_new = copy.deepcopy(win_wave)
        win_flux_new = copy.deepcopy(win_flux)
        # win_flux_err_new=copy.deepcopy(win_flux_err)
        for i in range(1, len(win_wave_new) - 1):
            wd_left = win_wave_new[i] - win_wave_new[i - 1]
            wd_right = win_wave_new[i + 1] - win_wave_new[i]
            fd_left = win_flux_new[i] - win_flux_new[i - 1]
            fd_right = win_flux_new[i + 1] - win_flux_new[i]
            rw = wd_left / (wd_left + wd_right)
            rf = fd_left / (fd_left + fd_right)

            if abs(rw - rf) > r0:
                win_wave.remove(win_wave_new[i])
                win_flux.remove(win_flux_new[i])
                # win_flux_err.remove(win_flux_err_new[i])

        if win_wave[win_flux.index(max(win_flux))] <= 4760:
            s0 = 0.015
        elif win_wave[win_flux.index(max(win_flux))] <= 5800:
            s0 = 0.02
        elif win_wave[win_flux.index(max(win_flux))] <= 6000:
            s0 = 0.03
        elif win_wave[win_flux.index(max(win_flux))] <= 7940:
            s0 = 0.03
        else:
            s0 = 0.05

        if len(win_flux) <= 29:
            c = 0.2
        elif len(win_flux) <= 59:
            c = 0.35
        elif len(win_flux) <= 89:
            c = 0.5
        elif len(win_flux) <= 119:
            c = 0.65
        elif len(win_flux) <= 149:
            c = 0.85
        else:
            c = 1.0

        # print(win_flux)
        spl = UnivariateSpline(win_wave, win_flux)
        spl.set_smoothing_factor(c * s0)
        flux_smooth = spl(wave[nn])
        # polate1=interpolate.interp1d(wave[nn], flux[nn], kind='linear')
        # flux_polate=polate1(wave_smooth)
        flux_nor = np.array(flux[nn]) / flux_smooth
        flux_nor_err = flux_nor * uncertainty[nn]
        for mm in range(len(wave[nn])):
            wave_total.append(wave[nn][mm])
            flux_total.append(flux[nn][mm])
            flux_smooth_total.append(flux_smooth[mm])
            flux_nor_total.append(flux_nor[mm])
            flux_nor_err_total.append(flux_nor_err[mm])

    # f10 = open(txt_dir + name + '.normalized.txt', 'w')
    # f10.write('wavelength flux flux_err' + '\n')
    # for i in range(len(flux_nor_total)):
    #     f10.write(str(wave_total[i]) + ' ' + str(flux_nor_total[i]) + ' ' + str(flux_nor_err_total[i]) + '\n')
    return wave_total, flux_nor_total


# -------------------------------------------------------------------------------------
wave_split=[5800.0, 8000.0, 9000.0]
obsid = []
fluxx = []
wavelengthh = []
rootdir = '/home/admin1/data1/czd/czd_lbw/159spec/'  # 根目录   #---------------spec_dir
list = os.listdir(rootdir)  # ------------spec_name
print(len(list))
#hdu=pd.read_csv("pos_classify_new_s_n_notin_old_s_n.csv")
#id=np.array(hdu['obsid'])
print("开始19")
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
        hdr = hdu[0].header
        if j >= 0 :
            try:
                # print('test3')
                orgwavelength = changeRV(hdu[1].data[0][2])  # 读数据，都加到数组orgwavelength里面
                orgflux = hdu[1].data[0][0]
            except:
                orgwavelength = changeRV(hdu[0].data[2])
                orgflux = hdu[0].data[0]
            aflux = []
            awave = []
            maxvalue = removeExpval(orgflux)
                # 总结就是maxvalue不是0的话，就不把最后一个值放进去
            if maxvalue != 0:  # max1>=1.5*max2
                    for f in range(len(orgflux)):
                        if orgflux[f] < maxvalue:
                            aflux.append(orgflux[f])
                            awave.append(orgwavelength[f])
            else:  # maxvalue==0
                    # awave = orgwavelength
                    # aflux = orgflux
                    awave,aflux,uncertain = uncertainty(rootdir, list[j], wave_split)

                    wave0, flux0 = normalize(awave, aflux, uncertain, wave_split)
                    # flux = (aflux - np.min(aflux)) / (np.max(aflux) - np.min(aflux))
                    if np.min(wave0) <= 4000:   
                           if np.max(wave0) >= 8000:                  
                               wavenew = np.linspace(4000, 8000, 3909)  # 用等差数列创造一组数
                           else:
                               wavenew = np.linspace(4000, 7999, 3909)  # 用等差数列创造一组数
                    else:
                           wavenew = np.linspace(4001, 8000, 3909)  # 用等差数列创造一组数
                    # linspace用于在线性空间中以均匀步长生成数字序列，通常用numpy.arange()生成序列，但是当我们使用浮点参数时，可能会导致精度损失
                    # 格式：array = numpy.linspace(start, end, num=num_points)将在start和end之间生成一个统一的序列，共有num_points个元素

                    f = interpolate.interp1d(wave0, flux0, kind="slinear")
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
                # n += 1
t = Table([obsid, wavelengthh, fluxx],
          names=('obsid', 'wavelength', 'flux'))
print(t.info)
t.write('/home/admin1/data1/czd/compare/159_laping.fits', format='fits', overwrite=True)
