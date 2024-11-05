import os

import numpy as np
import pandas as pd
import torch
from osgeo import gdal
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision.transforms import functional as F
from torch.nn import functional as f
from tqdm import tqdm

from funcs.functionds import depth_accuracy
from funcs.othernets import ResNet50_UNet, VGG16UNet, Eb3net, DeepLabV3_Regression
from funcs.upernet import UPerNet
from steste import find_folders_with_keyword
from unet import UNet


def get_file_paths(folder_path):
    file_paths = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        file_path = os.path.join(folder_path, filename)
        # 判断是否为文件（而不是子文件夹等）
        if os.path.isfile(file_path):
            file_paths.append(file_path)
    return file_paths


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
    if bandsnum>1:
        Tensor = F.to_tensor(data).permute(1, 0, 2)
    else:
        Tensor = F.to_tensor(data).permute(0, 2, 1)
    return Tensor


class CustomDataset64(Dataset):
    def __init__(self, image_paths1, image_paths2, target_paths):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.target_paths1 = target_paths

    def __len__(self):
        return len(self.image_paths1)

    def __getitem__(self, idx):
        image1 = ReadTif(self.image_paths1[idx])
        image1 = torch.where(image1<-50,-50,image1)
        image2 = ReadTif(self.image_paths2[idx])
        image2 = torch.where(image2<0,0,image2)
        target = ReadTif(self.target_paths1[idx])
        target = f.avg_pool2d(target, kernel_size=4, stride=4)
        return image1, image2, target


class CustomDataset256(Dataset):
    def __init__(self, image_paths1, image_paths2, target_paths1):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.target_paths1 = target_paths1

    def __len__(self):
        return len(self.image_paths1)

    def __getitem__(self, idx):
        image1 = ReadTif(self.image_paths1[idx])
        image1 = torch.where(image1<-50,-50,image1)
        image2 = ReadTif(self.image_paths2[idx])
        image2 = torch.where(image2<0,0,image2)
        target1 = ReadTif(self.target_paths1[idx])
        return image1, image2, target1


if __name__ == '__main__':
    pach_size = 256
    batch_size = 8
    learning_rate = 0.0005
    num_epochs = 200
    if torch.cuda.is_available():
        print('Quadro is here!')
        device = torch.device('cuda')
    else:
        print("no cuda")
        device = torch.device('cpu')

    '''别人的数据'''
    # # 定义文件路径
    # train_csv_path = r'D:\ky\cyx\datalist_china_train_0.7.csv'
    # test_csv_path = r'D:\ky\cyx\datalist_china_test_0.7.csv'
    #
    # # 定义路径前缀
    # s1path = r'D:\ky\cyx\s1china_check'
    # # s2path = r'D:\ky\cyx\s2china_check'
    # s2path = r'D:\ky\cyx\gechina_check'
    # targetpath = r'D:\ky\cyx\bhchina'
    #
    # # 读取CSV文件
    # train_df = pd.read_csv(train_csv_path)
    # test_df = pd.read_csv(test_csv_path)
    #
    # # 获取文件名称列表
    # train_files = train_df.iloc[:, 0].tolist()
    # test_files = test_df.iloc[:, 0].tolist()
    #
    # # 生成完整路径
    # s1_train_paths = [f"{s1path}\\{filename}" for filename in train_files]
    # s2_train_paths = [f"{s2path}\\{filename}" for filename in train_files]
    # target_train_paths = [f"{targetpath}\\{filename}" for filename in train_files]
    #
    # s1_test_paths = [f"{s1path}\\{filename}" for filename in test_files]
    # s2_test_paths = [f"{s2path}\\{filename}" for filename in test_files]
    # target_test_paths = [f"{targetpath}\\{filename}" for filename in test_files]
    #
    # dateset_train = CustomDataset256(s1_train_paths, s2_train_paths, target_train_paths)
    # dateset_test = CustomDataset256(s1_test_paths, s2_test_paths, target_test_paths)
    # train_loader = DataLoader(dateset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(dateset_test, batch_size=batch_size, shuffle=False, drop_last=True)
    '''别人的数据'''

    '''自己的数据'''
    data_path = './data4'
    data_paths = [name for name in os.listdir(data_path)]

    dataset_all = []
    for i, path in enumerate(data_paths):
        current_path = os.path.join(data_path, path)
        current_path_s1 = os.path.join(current_path, 'sentinel1_AREA')
        image_paths_s1 = get_file_paths(current_path_s1)
        current_path_s2 = os.path.join(current_path, 'sentinel2_AREA')
        image_paths_s2 = get_file_paths(current_path_s2)
        current_path_bhrf = find_folders_with_keyword(current_path, 'bhr')
        target_paths = get_file_paths(current_path_bhrf[0])
        # current_path_cbra = os.path.join(current_path, 'cbra2018')
        # cbra_paths = get_file_paths(current_path_cbra)
        dataset1 = CustomDataset256(image_paths_s1, image_paths_s2, target_paths)
        dataset_all.append(dataset1)
    dataset = ConcatDataset(dataset_all)

    print('总数  ', len(dataset))
    train_ratio = 0.7  # 80% 的数据作为训练集
    test_ratio = 1 - train_ratio

    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=test_ratio, random_state=65)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    print('训练  ', len(train_dataset))
    print('测试  ', len(test_dataset))
    # 创建 DataLoader 加载训练集和测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    '''自己的数据'''

    # model = UNet(feature_scale=4,in_ch=5,out_ch=1)
    # model = ResNet50_UNet(in_channels=5)
    # model = VGG16UNet(in_channels=5)
    model = UPerNet(1)
    # model = Eb3net(num_channels=6, num_classes=1)
    # model = DeepLabV3_Regression(in_channels=5)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_rmse = float('inf')  # 初始化最好的RMSE

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs} training...')

        # 模型训练阶段
        model.train()
        train_loss = 0.0

        for data in tqdm(train_loader):  # tqdm可选，用于显示进度条
            inputs1, inputs2, targets = data
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            targets = targets.to(device)
            # upsampled_s1 = f.interpolate(inputs1, size=(256, 256), mode='bilinear', align_corners=False)
            # inputs = torch.cat((upsampled_s1, inputs2), dim=1)
            inputs = torch.cat((inputs1, inputs2), dim=1)
            # 前向传播
            outputs = model(inputs)
            loss = f.mse_loss(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            train_loss += loss.item()

        # 计算平均训练损失
        train_loss /= len(train_loader)
        print(f'Train Loss: {train_loss:.4f}')

        # 模型验证/测试阶段
        model.eval()  # 切换模型到评估模式
        test_loss = 0.0
        test_losa = 0.0
        test_d1 = 0.0
        test_d2 = 0.0
        test_d3 = 0.0
        doc = 0
        with torch.no_grad():  # 关闭梯度计算
            for data in tqdm(test_loader):
                inputs1, inputs2, targets = data
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                targets = targets.to(device)
                # upsampled_s1 = f.interpolate(inputs1, size=(256, 256), mode='bilinear', align_corners=False)
                # inputs = torch.cat((upsampled_s1, inputs2), dim=1)
                inputs = torch.cat((inputs1, inputs2), dim=1)
                outputs = model(inputs)
                loss = f.mse_loss(outputs, targets)
                losa = f.l1_loss(outputs, targets)

                mask2 = targets > 0
                if mask2.any():
                    derta1, derta2, derta3 = depth_accuracy(outputs, targets, mask2)
                    test_d1 += derta1
                    test_d2 += derta2
                    test_d3 += derta3
                else:
                    doc += inputs1.size(0)

                test_loss += loss.item()
                test_losa += losa.item()

        # 计算平均测试损失
        test_loss /= len(test_loader)
        test_losa /= len(test_loader)
        test_d1 = test_d1 / (len(test_loader) - doc)
        test_d2 = test_d2 / (len(test_loader) - doc)
        test_d3 = test_d3 / (len(test_loader) - doc)
        print(f'Test mse Loss: {test_loss:.4f}')
        print(f'Test mae Loss: {test_losa:.4f}')
        print(f'Test d1 Loss: {test_d1:.4f}')
        print(f'Test d2 Loss: {test_d2:.4f}')
        print(f'Test d3 Loss: {test_d3:.4f}')

        # 计算 RMSE
        rmse = torch.sqrt(torch.tensor(test_loss))
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {test_losa:.4f}')

        # 保存最优模型
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), 'models/upernet.pth')
            print(f'Best model saved with RMSE: {best_rmse:.4f}')

    print('Training and testing complete.')