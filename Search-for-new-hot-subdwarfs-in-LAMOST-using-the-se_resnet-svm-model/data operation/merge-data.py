import os
import gzip
import shutil

def decompress_fits_gz(source_directory, target_directory):
    # 确保目标目录存在，如果不存在，则创建它
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 搜索目录中所有的 .fits.gz 文件
    for file in os.listdir(source_directory):
        if file.endswith(".fits.gz"):
            # 构建完整的源文件路径
            source_file_path = os.path.join(source_directory, file)
            # 构建目标文件路径（移除 .gz）
            target_file_path = os.path.join(target_directory, file[:-3])

            # 解压文件到目标目录
            with gzip.open(source_file_path, 'rb') as f_in:
                with open(target_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            print(f"File decompressed: {target_file_path}")

# 使用示例
source_directory = '/home/admin1/data/mjq/3066/'  # 源文件夹路径
target_directory = '/home/admin1/data/mjq/3066ex'  # 目标文件夹路径
decompress_fits_gz(source_directory, target_directory)
