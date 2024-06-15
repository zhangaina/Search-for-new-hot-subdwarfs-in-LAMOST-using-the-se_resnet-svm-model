from astropy.io import fits
from astropy.table import Table
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def merge_fits_files(directory, output_file):
    first_file = True
    merged_data = None

    # 获取目录中所有的 .fits 文件
    fits_files = list(Path(directory).rglob('*.fits'))
    
    # 使用 tqdm 显示进度
    for file in tqdm(fits_files, desc='Processing files', unit='file'):
        if file.name == "WDdata.fits":
            continue
        with fits.open(file) as hdu:
            data = hdu[1].data  # 假设数据存储在第二个 HDU 中
            obsid = data['obsid']
            wavelength = data['wavelength']
            flux = data['flux']

            # 创建一个新的表格，包含提取的参数
            table = Table([obsid, wavelength, flux], names=('obsid', 'wavelength', 'flux'))

            if first_file:
                merged_data = table
                first_file = False
            else:
                table_data = table
                merged_data = np.concatenate((merged_data, table_data), axis=0)


    # 创建一个新的 Binary Table HDU 对象
    merged_table = Table(merged_data, names=('obsid', 'wavelength', 'flux'))
    hdu = fits.BinTableHDU(data=merged_table)

    # 创建 HDU list，并写入到新的 .fits 文件
    hdu_list = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdu_list.writeto(output_file, overwrite=True)

if __name__ == '__main__':
    directory = '/home/admin1/data/mjq/HSD/negatives'
    output_file = '/home/admin1/data/mjq/merged_data_noWD.fits'
    merge_fits_files(directory, output_file)
