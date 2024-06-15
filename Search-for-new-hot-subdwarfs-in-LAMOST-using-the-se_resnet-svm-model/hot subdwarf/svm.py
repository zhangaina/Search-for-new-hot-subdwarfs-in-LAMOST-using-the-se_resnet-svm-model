from sklearn import svm
import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV

#训练svm进行分类



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
    return returnflux

if __name__=='__main__':

    """filepath1 = r'/home/admin1/data/mjq/HSD/extract/ex2905_noWD.fits'
    filepath2 = r'/home/admin1/data/mjq/HSD/extract/exnegatives_test_noWD.fits'
    filepath3 = r'/home/admin1/data/mjq/HSD/extract/exnegatives_train_noWD.fits'"""

    filepath1 = r'/home/admin1/data1/czd/two-class-model/extract/exhots_WD.fits'
    filepath2 = r'/home/admin1/data1/czd/two-class-model/extract/exnegatives_test_WD1000.fits'
    filepath3 = r'/home/admin1/data1/czd/two-class-model/extract/exnegatives_train_WD1000.fits'
    # newHot = 'mixbhbnewHot.fits'
    newHot = filepath1
    
    # bhb = 'mixbhb.fits'

    hdu = fits.open(newHot)
    old_obsid = []
    old_flux = []
    all_data = []
    all_label = []
    tdata = []
    fdata = []
    for i in range(len(hdu[1].data)):
        old_obsid.append(hdu[1].data[i]['obsid'])
        temp = hdu[1].data[i]['flux']
        # print(len(cutflux(waveleng, hdu[1].data[i]['flux'][0:3909])))
        all_data.append(temp)
        old_flux.append(temp)
        all_label.append(0)

        # 增加正样本数量
        # all_data.append(temp)
        # all_label.append(0)

    length = len(all_data)

    # hdu = fits.open(bhb)
    # for i in range(length):
    #     temp = hdu[1].data[i]['flux'][3909:]
    #     temp = np.append(cutflux(waveleng, hdu[1].data[i]['flux'][0:3909]), temp)
    #     all_data.append(temp)
    #     all_label.append(1)
    print(all_data[0])

    train_x,test_x, train_y,test_y = train_test_split(all_data, all_label, test_size=0.3)

    hdu_negative_test = fits.open(filepath2)
    for i in range(len(hdu_negative_test[1].data)):
        temp = hdu_negative_test[1].data[i]['flux']
        train_x.append(temp)
        train_y.append(1)

    hdu_negative_train = fits.open(filepath3)
    for i in range(len(hdu_negative_train[1].data)):
        temp = hdu_negative_train[1].data[i]['flux']
        test_x.append(temp)
        test_y.append(1)
    #print(len(hdu_negative_train[1].data[i]['flux']))
    
    
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000, 10000, 100000],  # C参数的值
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],  # gamma参数的值
        'kernel': ['rbf']  # 使用径向基函数（RBF）核
    }

    # 创建SVC模型
    classifier = svm.SVC()

    # 创建GridSearchCV实例
    grid_search = GridSearchCV(classifier, param_grid, refit=True, verbose=2, cv=5)

    """# 进行网格搜索
    grid_search.fit(train_x, train_y)#如果要重新计算参数那么要加上

    # 输出最佳参数
    print("最佳参数：", grid_search.best_params_)
    """

    classifier = svm.SVC(C=10, kernel='rbf', gamma=0.01, decision_function_shape='ovr', probability=1)
    classifier.fit(train_x, train_y)
        

    # y_pred = classifier.predict_proba(test_x)
    # cm = confusion_matrix(test_y, y_pred)
    # cm = confusion_matrix(test_y, classifier.predict(test_x))
    y_pred = classifier.predict(test_x)

    print('测试集acc为：{0:.3f}%'.format(accuracy_score(test_y, y_pred) * 100))
    print('测试集f1_score为：{0:.3f}%'.format(f1_score(test_y, y_pred) * 100))
    print('测试集recall_score为：{0:.3f}%'.format(recall_score(test_y, y_pred) * 100))
    print('测试集precision_score为：{0:.3f}%'.format(precision_score(test_y, y_pred) * 100))


    
#------------------------------------------------------------
    
    # dir = "/home/admin1/MJQ/output/test2.fits" # 拉平后的3163
    dir = r'/home/admin1/data/mjq/HSD/extract/ex31632.fits' # 特征提取
    hdu = fits.open(dir)
    new_data = []
    new_obsid_all = []

    same_cnt = 0

    for i in range(len(hdu[1].data)):
        new_obsid_all.append(hdu[1].data[i]['obsid'])
        if hdu[1].data[i]['obsid'] in old_obsid:
            same_cnt += 1
        temp = hdu[1].data[i]['flux']
        new_data.append(temp)

    new_pred = classifier.predict_proba(new_data)
    # cm = confusion_matrix(test_y, classifier.predict(test_x))
    # y_pred = classifier.predict(test_x)

    new_hot_cnt = 0
    new_obsid = []

    for i in range(len(new_pred)):
        
        if new_pred[i][0] > 0.5:
            # print(new_pred[i][0], new_pred[i][1])
            new_obsid.append(hdu[1].data[i]['obsid'])
            new_hot_cnt += 1

    print("total_find:", new_hot_cnt, "/3163")
    new_find_cnt = new_hot_cnt

    for i in range(len(new_obsid)):
        if new_obsid[i] in old_obsid:
            new_find_cnt -= 1

    print("old_find:", new_hot_cnt - new_find_cnt, "/", same_cnt)

    print("new_find:", new_find_cnt, "/", 3163 - same_cnt)

    # for i in range(len(new_obsid_all)):
    #         if new_obsid_all[i] in old_obsid:
    #             index_in_old = old_obsid.index(new_obsid_all[i])
    #             flux1 = old_flux[index_in_old]
    #             flux2 = new_data[i]
    #             print("stop")

    find_cnt = 176

    file_path = '/home/admin1/data/mjq/sd_mpfit_parameter.fits'
    hdu_176 = fits.open(file_path)

    filtered_data = []

    for i in range(len(hdu_176[1].data)):
        if hdu_176[1].data[i]['obsid'] not in new_obsid:
            filtered_data.append(hdu_176[1].data[i])
            find_cnt -= 1

    print("176 find :", find_cnt, " / 176")
    original_columns = hdu_176[1].columns

    find_176 = 0

    for i in range(len(hdu_176[1].data)):
        if hdu_176[1].data[i]['obsid'] in new_obsid_all:
            find_176 += 1
    
    print("find_176: ", find_176)

# 使用原始列定义和筛选后的数据创建新的 FITS 表
    new_hdu = fits.BinTableHDU.from_columns(original_columns, nrows=len(filtered_data))
    for i, row in enumerate(filtered_data):
        for colname in original_columns.names:
            new_hdu.data[colname][i] = row[colname]


    # 保存新的 FITS 文件
    new_file_path = '/home/admin1/data1/czd/output/notfind+WD10002.fits'  # 指定新文件的路径和文件名
    new_hdu.writeto(new_file_path, overwrite=True)
