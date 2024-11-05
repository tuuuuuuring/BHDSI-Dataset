import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from torchvision.transforms import functional as F


def ReadTif(image_path):
    dataset = gdal.Open(image_path)
    bandsnum = dataset.RasterCount
    # 获得矩阵的列数
    width = dataset.RasterXSize
    # 栅格矩阵的行数
    height = dataset.RasterYSize
    # 获得数据
    data = dataset.ReadAsArray(0, 0, width, height)
    data = np.array(data).astype(np.float32)
    # print(data.shape)
    # vvstd = 5.894991953747706
    # meanvv = -9.400159472318535
    # vhstd = 6.5062821612476816
    # meanvh = -17.48087407519381
    # b2std = 0.041234904196493866
    # meanb2 = 0.09032810650349908
    # b3std = 0.044207560644641535
    # meanb3 = 0.11008062839901452
    # b4std = 0.058242708431953054
    # meanb4 = 0.11641374186227577
    # b8std = 0.07429693760337167
    # meanb8 = 0.20418531308997234
    # if bandsnum == 2:
    #     with rasterio.open(image_path) as src:
    #         band_vv=src.read(1)
    #         band_vv=(band_vv-meanvv)/vvstd
    #         band_vh=src.read(2)
    #         band_vh = (band_vh - meanvh) / vhstd
    #     stacked_bands = np.stack([band_vv,band_vh],axis=0)
    #     Tensor = torch.tensor(stacked_bands, dtype=torch.float32)
    # elif bandsnum ==4:
    #     with rasterio.open(image_path) as src:
    #         band_b2 = src.read(1)
    #         band_b2 = (band_b2-meanb2)/b2std
    #         band_b3 = src.read(2)
    #         band_b3 = (band_b3 - meanb3) / b3std
    #         band_b4 = src.read(3)
    #         band_b4 = (band_b4 - meanb4) / b4std
    #         band_b8 = src.read(4)
    #         band_b8 = (band_b8 - meanb8) / b8std
    #     stacked_bands = np.stack([band_b2,band_b3,band_b4,band_b8],axis=0)
    #     Tensor = torch.tensor(stacked_bands, dtype=torch.float32)
    if bandsnum>1:
        Tensor = F.to_tensor(data).permute(1, 0, 2)
    else:
        Tensor = F.to_tensor(data).permute(0, 2, 1)
    return Tensor


if __name__ == '__main__':
    # 定义文件路径
    train_csv_path = r'D:\ky\cyx\datalist_china_train_0.7.csv'
    test_csv_path = r'D:\ky\cyx\datalist_china_test_0.7.csv'

    # 定义路径前缀
    s2path = r'D:\ky\gechina_check'
    targetpath = r'D:\ky\cyx\bhchina'

    # 读取CSV文件
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # 获取文件名称列表
    train_files = train_df.iloc[:, 0].tolist()
    test_files = test_df.iloc[:, 0].tolist()

    # 生成完整路径
    s2_train_paths = [f"{s2path}\\{filename}" for filename in train_files]
    target_train_paths = [f"{targetpath}\\{filename}" for filename in train_files]

    s2_test_paths = [f"{s2path}\\{filename}" for filename in test_files]
    target_test_paths = [f"{targetpath}\\{filename}" for filename in test_files]

    maxheight = 0.0
    for path in target_train_paths:
        with rasterio.open(path) as src:
            data = src.read(1)
            maxvalue = np.max(data)
            if maxvalue>maxheight:
                maxheight = maxvalue
    for path in target_test_paths:
        with rasterio.open(path) as src:
            data = src.read(1)
            maxvalue = np.max(data)
            if maxvalue > maxheight:
                maxheight = maxvalue

    print(maxheight)
