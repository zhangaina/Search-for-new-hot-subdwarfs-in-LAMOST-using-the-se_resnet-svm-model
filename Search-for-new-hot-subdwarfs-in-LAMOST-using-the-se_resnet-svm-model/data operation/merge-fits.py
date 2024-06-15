from astropy.io import fits
from astropy.table import Table, hstack
import numpy as np

def merge_fits(file1, file2):
    # 打开两个FITS文件
    with fits.open(file1) as hdu1, fits.open(file2) as hdu2:
        # 获取数据
        data1 = hdu1[1].data
        data2 = hdu2[1].data

        set_tmp = set()
        
        for i in range(len(data1)):
            set_tmp.add(data1[i][0])

        for i in range(len(data2)):
            print("i = ", i)
            while data2[i][0] in set_tmp:
                data2 = np.delete(data2, i, axis=0)
                if i >= len(data2):
                    break
            if i >= len(data2):
                break

        tmp = np.concatenate((data1, data2), axis=0)

        hdu1[1].data = tmp
        hdu1.writeto('/home/admin1/data/mjq/merge.fits', overwrite=True)

        

if __name__ == '__main__':
    file1 = '/home/admin1/data/mjq/HSD/merge.fits'
    file2 = '/home/admin1/data/mjq/newHot.fits'

    merge_fits(file1, file2)

    # hdu = fits.open(file1)
    # data = hdu[1].data
    # cols = hdu[1].columns
    # new2290 = fits.BinTableHDU.from_columns([cols[0], cols[2], cols[3]])
    # new2290.writeto(output_file, overwrite=True)

