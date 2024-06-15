from pylamost import lamost
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

def download_lamost_data1(lmo, data_list):
    for i in tqdm(data_list):
            lmo.downloadFits(obsid=i, savedir='/home/admin1/data1/czd/czd_lbw/159spec')

if __name__ == '__main__':
    lm = lamost(dataset=8, version=1.0)  # init the lamost class
    lm.token = 'F0aedc160fb'

    underdownload_file_list = '159.csv'
    download_list = pd.read_csv(
        underdownload_file_list, usecols=['obsid'], index_col=None)

    # print(download_list)
    data_array = np.array(download_list)
    data_array = data_array.reshape(-1, )
    all_data_list = data_array.tolist()

    print(len(all_data_list))

    # n可以改，根据每次停下的位置
    #n = 31966
    gap=330000#区间间隔
    gap_final=254897#最后一个区间间隔
    totalgap = 67920
    # offset 是区分组的，不要改
    #offset = 300000
    # download_lamost_data(lm,all_data_list[0:n])
    process_pool = []
    # for i in range(20):
        # p = Process(target=download_lamost_data, args=(
        #     lm, all_data_list[n + offset*i:offset*(i+1)]))
        # process_pool.append(p)
    rootdir1 = '/home/admin1/data1/czd/czd_lbw/159spec/'  # 根目录

    list1 = os.listdir(rootdir1)

    print(len(list1))
    
    p1 = Process(target=download_lamost_data1, args=(lm, all_data_list[0 + len(list1):totalgap]))

    process_pool.append(p1)

    
    for pp in process_pool:
        pp.start()
    for pp in process_pool:
        pp.join()

    print("运行结束")

