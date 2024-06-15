from astropy.io import fits
import pandas as pd

# FITS 文件路径
fits_file_path = '/home/admin1/data/mjq/HSD/hots.fits'

# 读取 FITS 文件
hdu = fits.open(fits_file_path)

# 获取数据表（假设数据在第一个扩展中）
data_table = hdu[1].data

obsid = data_table['obsid']

# 创建 Pandas DataFrame
df = pd.DataFrame({'obsid': obsid})

# 将 DataFrame 保存为 CSV 文件
csv_file_path = '/home/admin1/data/mjq/HSD/hots_obsid.csv'
df.to_csv(csv_file_path, index=False)

# 打印成功保存的消息
print(f"数据已保存至 {csv_file_path}")

# 关闭 FITS 文件
hdu.close()
