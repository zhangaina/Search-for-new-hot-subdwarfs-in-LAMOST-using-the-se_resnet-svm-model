from astropy.io import fits
from astropy.table import Table
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def merge_and_split_fits_files(directory, output_train_file, output_test_file):
    first_file = True
    merged_data = None

    # 获取目录中所有的 .fits 文件
    fits_files = list(Path(directory).rglob('*.fits'))

    #fits_files = [file for file in fits_files if file.name != "WDdata.fits"]
    
    # 使用 tqdm 显示进度
    for file in tqdm(fits_files, desc='Processing files', unit='file'):
        with fits.open(file) as hdu:
            data = hdu[1].data  # 假设数据存储在第二个 HDU 中
            obsid = data['obsid']
            wavelength = data['wavelength']
            flux = data['flux']

            # 创建一个新的表格，包含提取的参数
            table = Table([obsid, wavelength, flux], names=('obsid', 'wavelength', 'flux'))

            if len(data) > 2000:
                indices = np.random.choice(len(data), 2000, replace=False)
                sample = table[indices]
            else:
                sample = table


            split_index = int(len(sample) * 1)
            train_sample = sample[:split_index]
            
            if first_file:
                train_data = train_sample
                first_file = False
            else:
                train_data = np.concatenate((train_data, train_sample), axis=0)
                
    train_table = Table(train_data, names=('obsid', 'wavelength', 'flux'))
    train_hdu = fits.BinTableHDU(data=train_table)
    train_hdu_list = fits.HDUList([fits.PrimaryHDU(), train_hdu])
    train_hdu_list.writeto(output_train_file, overwrite=True)

if __name__ == '__main__':
    directory = '/home/admin1/data/LBW/czd/A'
    output_train_file = '/home/admin1/data/LBW/czd/A/A.fits'
    merge_and_split_fits_files(directory, output_train_file, output_test_file)
