# This is a sample Python script.
import os
import pickle
import random
from types import SimpleNamespace

import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from osgeo import gdal

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import ConcatDataset
from alive_progress import alive_bar

from torch.nn import functional as f
import numpy as np

from funcs.caisdataset import get_dataset
from funcs.datasets import CustomDataset, CustomDataset64, CustomDatasetGoogle
from funcs.dpt import DPT
from funcs.evaluator import LOSS_OR, LOSS_OR_2, LOSS_OR_3, REL
from funcs.functionds import calculate_iou, relative_absolute_error, depth_accuracy, r2_score
from funcs.upernet import UPerNet
from networks import original
from newcrfs.networks.NewCRFDepth import NewCRFDepthwithuper
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


def get_data_in(dp):
    data_path = dp
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
        current_path_cbra = os.path.join(current_path, 'cbra2018')
        cbra_paths = get_file_paths(current_path_cbra)
        dataset1 = CustomDataset(image_paths_s1, image_paths_s2, target_paths, cbra_paths)
        dataset_all.append(dataset1)
    return dataset_all


def random_flip(input1, input2, output1, output2):
    """
    对输入张量进行随机水平和垂直翻转。

    参数:
        tensor (torch.Tensor): 形状为 (batch_size, channels, height, width) 的张量
        p (float): 进行翻转的概率，默认为0.5

    返回:
        torch.Tensor: 翻转后的张量
    """
    if 3*torch.rand(1).item() < 1:
        input1 = torch.flip(input1, [3])  # 水平翻转 (width 维度)
        input2 = torch.flip(input2, [3])
        output1 = torch.flip(output1, [3])  # 水平翻转 (width 维度)
        output2 = torch.flip(output2, [3])
    elif 1 < 3*torch.rand(1).item() < 2:
        input1 = torch.flip(input1, [2])  # 垂直翻转 (width 维度)
        input2 = torch.flip(input2, [2])
        output1 = torch.flip(output1, [2])  # 垂直翻转 (width 维度)
        output2 = torch.flip(output2, [2])
    return input1, input2, output1, output2


def paint_tests(outputs, targets,epcho,losser):
    pic_save_dir = "./pic"
    pic_save_dirc = os.path.join(pic_save_dir, f"{epcho+1}lun")

    # 检查是否已存在指定 epoch 的文件夹，如果不存在则创建
    if not os.path.exists(pic_save_dirc):
        os.makedirs(pic_save_dirc)

    # 根据 high_or_low 参数决定创建 high 或 low 文件夹
    subdir_name = f"mse{losser}"

    subdir_path = os.path.join(pic_save_dirc, subdir_name)

    # 检查是否已存在指定的 high 或 low 文件夹，如果不存在则创建
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)
    outputs_cpu = outputs.cpu()
    targets_cpu = targets.cpu()
    outputs_np = outputs_cpu.numpy()
    targets_np = targets_cpu.numpy()

    # 计算所有图像的全局最小值和最大值
    global_min = min(np.min(outputs_np[0]), np.min(targets_np[0]))
    global_max = max(np.max(outputs_np[0]), np.max(targets_np[0]))

    # 绘制输出均值图像
    plt.figure(figsize=(8, 6))
    plt.imshow(np.mean(outputs_np[0], axis=0), cmap='viridis', vmin=global_min, vmax=global_max)
    plt.colorbar()
    output_path = os.path.join(subdir_path, "output_mean.png")
    plt.savefig(output_path)
    plt.close()

    # 绘制目标均值图像
    plt.figure(figsize=(8, 6))
    plt.imshow(np.mean(targets_np[0], axis=0), cmap='viridis', vmin=global_min, vmax=global_max)
    plt.colorbar()
    target_path = os.path.join(subdir_path, "target_mean.png")
    plt.savefig(target_path)
    plt.close()


def Loss_mse_cs(hp, hr, logvar):
    percision = torch.exp(-logvar)
    l1 = f.mse_loss(hp, hr, reduction='mean')
    cs = f.cosine_similarity(hp, hr)
    l2 = percision * l1 + logvar
    return torch.mean(cs) + l2


def Loss_mse_bce3(hp, hr, lovar):
    pecision = torch.exp(-lovar)
    loss0 = f.mse_loss(hp[0], hr[0], reduction='mean')
    diff = f.cross_entropy(hp[1], hr[1], reduction='mean')
    loss1 = pecision*diff+lovar
    return loss0 + loss1


def Loss_mse_dicebce(hp, hr, lovar):
    pecision = torch.exp(-lovar)
    loss0 = f.smooth_l1_loss(hp[0], hr[0], reduction='mean')
    diff = f.binary_cross_entropy(hp[1], hr[1])
    loss1 = pecision*diff+lovar
    return loss0 + loss1


def Loss_mse_bce(hp, hr, lovars):
    loss1 = f.cross_entropy(hp[0], hr[1], reduction='mean')
    persision1 = torch.exp(lovars)
    diff1 = f.mse_loss(hp[1], hr[0], reduction='mean')
    loss2 = persision1 * diff1 + lovars

    return loss1 + loss2


def Loss_mse2_bce(hp, hr):
    # persision0 = torch.exp(-lovars[0])
    diff0 = f.mse_loss(hp[1], hr[0], reduction='mean')+0.5*f.mse_loss(hp[2], hr[0], reduction='mean')
    # loss0 = persision0 * diff0 + lovars[0]
    loss1 = f.cross_entropy(hp[0], hr[1], reduction='mean')

    return 0.8*diff0 + 0.2*loss1


def test_epoch(model, testloader, device,epcho):
    test_loss = 0.0
    test_loss_rel = 0.0
    test_d1 = 0.0
    test_d2 = 0.0
    test_d3 = 0.0
    test_iou = 0.0
    cvbd=0
    test_r2=0.0
    doc=0

    random_indices = random.sample(range(len(testloader)), 3)
    model.eval()
    with torch.no_grad():  # 不进行梯度计算
        with alive_bar(len(test_loader), force_tty=True) as par:
            for tobtest in testloader:
                images1, images2, targets, cbra = tobtest
                images1 = images1.to(device)
                images2 = images2.to(device)
                targets = targets.to(device)
                cbra = cbra.to(device)

                outputs1 = model(images1, images2)

                mskbh = (targets > 0)
                mskuis = (targets == 0) & (cbra == 0)
                msk = mskuis | mskbh
                mask_bh_masked = targets[msk]
                pred_bh_masked = outputs1[msk]
                iou=1

                mask2 = targets > 0
                if mask2.any():
                    derta1,derta2,derta3=depth_accuracy(outputs1,targets,mask2)
                    test_d1 += derta1 * images1.size(0)
                    test_d2 += derta2 * images1.size(0)
                    test_d3 += derta3 * images1.size(0)
                else:
                    doc+=images1.size(0)

                # losst = f.mse_loss(outputs1, targets, reduction='mean').cpu().detach().numpy()
                losst =f.mse_loss(pred_bh_masked, mask_bh_masked, reduction='mean').cpu().detach().numpy()

                rel_loss = relative_absolute_error(outputs1,targets,mask2)

                r2 = r2_score(outputs1,targets)

                test_loss_rel += rel_loss.item() * images1.size(0)
                test_loss += losst.item() * images1.size(0)
                test_r2 += r2.item() * images1.size(0)
                test_iou += iou * images1.size(0)

                if epcho%5==0:
                    if cvbd in random_indices:
                        paint_tests(outputs1, targets, epcho, losst)
                    # if ((not low_have_painted) and losst<40) or ((not high_have_painted) and losst>70):
                    #     if losst<40:
                    #         hol=False
                    #         low_have_painted=True
                    #     else:
                    #         high_have_painted=True
                    #         hol=True
                    #     paint_tests(outputs, targets,epcho,hol)

                cvbd+=1

                par.text('mse %.4f,r2 %.4f, iou %.4f' % (losst.item(), r2.item(), iou))
                par()
            test_loss = np.sqrt(test_loss / len(testloader.dataset))
            test_loss_r2 = test_r2 / len(testloader.dataset)
            test_rel = test_loss_rel/ len(testloader.dataset)
            test_d1 = test_d1/(len(testloader.dataset)-doc)
            test_d2 = test_d2/(len(testloader.dataset)-doc)
            test_d3 = test_d3/(len(testloader.dataset)-doc)
            test_iou = test_iou/len(testloader.dataset)
            print(f"Test rmse Loss: {test_loss:.4f}")
            print(f"Test rel: {test_rel:.4f}")
    return [test_loss, test_loss_r2, test_d1, test_d2, test_d3, test_rel], ['rmse', 'r2', 'd1', 'd2', 'd3', 'rel']


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    batch_size = 8
    learning_rate = 0.0005
    num_epochs = 200
    if torch.cuda.is_available():
        print_hi('Quadro is here!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("no cuda")
    save_dir = "./models"
    '''new datas'''

    # argsd = SimpleNamespace(data_path=r'D:\paisenPR\bhr_by_dl\data4\cut1',
    #                         train_index='../data_path/ntraining_data_cbw_30.csv',
    #                         val_index='../data_path/nvalidation_data_cbw.csv',
    #                         s1_band=2,
    #                         s2_band=4,
    #                         phase='train',
    #                         expand_index='None',
    #                         uis_name='wsf',
    #                         data_name='cai'
    #                         )
    # datasets = []
    # for i in range(67):
    #     argsd.data_path = os.path.join(r'D:\paisenPR\bhr_by_dl\data4', f'cut{i + 1}')
    #     train_dataset = get_dataset(argsd)
    #     datasets.append(train_dataset)
    # datasets = ConcatDataset(datasets)
    # train_ratio = 0.7  # 70% 的数据作为训练集
    # test_ratio = 1 - train_ratio
    #
    # # 使用 train_test_split 函数划分数据集
    # train_indices, test_indices = train_test_split(list(range(len(datasets))), test_size=test_ratio, random_state=84)
    # train_dataset = Subset(datasets, train_indices)
    # val_dataset = Subset(datasets, test_indices)
    # print(' len train dataset', str(len(train_dataset)))
    # print(' len test dataset', str(len(val_dataset)))
    #
    # train_loader = DataLoader(train_dataset, shuffle=True, num_workers=4,
    #                         batch_size=batch_size, drop_last=True, pin_memory=False)
    # test_loader = DataLoader(val_dataset, shuffle=False, num_workers=4,
    #                         batch_size=4, drop_last=True, pin_memory=False)
    '''new data'''

    '''old data'''
    # dataset_all = get_data_in('./data')
    # dataset = ConcatDataset(dataset_all)
    #
    # print('总数  ', len(dataset))
    # train_ratio = 0.75  # 80% 的数据作为训练集
    # test_ratio = 1 - train_ratio
    #
    # # 使用 train_test_split 函数划分数据集
    # # with open('train_test_indices.pkl', 'wb') as fff:
    # #     pickle.dump((train_indices, test_indices), fff)
    #
    # train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=test_ratio, random_state=65)
    # train_dataset = Subset(dataset, train_indices)
    # test_dataset = Subset(dataset, test_indices)
    # print('训练  ', len(train_dataset))
    # print('测试  ', len(test_dataset))
    # # 创建 DataLoader 加载训练集和测试集
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    '''old data'''

    '''别人的数据'''
    # 定义文件路径
    train_csv_path = r'D:\ky\cyx\datalist_china_train_0.7.csv'
    test_csv_path = r'D:\ky\cyx\datalist_china_test_0.7.csv'

    # 定义路径前缀
    s1path = r'D:\ky\cyx\s1china_check'
    s2path = r'D:\ky\cyx\gechina_check'
    targetpath = r'D:\ky\cyx\bhchina'

    # 读取CSV文件
    train_df = pd.read_csv(train_csv_path, header=None)
    test_df = pd.read_csv(test_csv_path, header=None)

    # 获取文件名称列表
    train_files = train_df.iloc[:, 0].tolist()
    test_files = test_df.iloc[:, 0].tolist()

    # 生成完整路径
    s1_train_paths = [f"{s1path}\\{filename}" for filename in train_files]
    s2_train_paths = [f"{s2path}\\{filename}" for filename in train_files]
    target_train_paths = [f"{targetpath}\\{filename}" for filename in train_files]

    s1_test_paths = [f"{s1path}\\{filename}" for filename in test_files]
    s2_test_paths = [f"{s2path}\\{filename}" for filename in test_files]
    target_test_paths = [f"{targetpath}\\{filename}" for filename in test_files]

    dateset_train = CustomDatasetGoogle(s1_train_paths, s2_train_paths, target_train_paths)
    dateset_test = CustomDatasetGoogle(s1_test_paths, s2_test_paths, target_test_paths)
    print('训练  ', len(dateset_train))
    print('测试  ', len(dateset_test))
    train_loader = DataLoader(dateset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dateset_test, batch_size=batch_size, shuffle=False, drop_last=True)
    '''别人的数据'''

    '''模型'''
    # model = separate_way(block='bt', fusion='sknet')
    # model = DualUnetnocat(block='bt')
    # model = little_mdf(block='bt', fusion='sknet')
    # model = original(block='bt', fusion='sknet',in_ch1=2,in_ch2=4)
    model = NewCRFDepthwithuper(version='tiny07')
    # model = UPerNet()
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        print_hi('more than one quadro')
    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    '''参数'''
    log_var = torch.zeros((1,), requires_grad=True)
    # log_vars = torch.zeros((1,), requires_grad=True)
    # log_varss = torch.zeros((1,), requires_grad=True)

    # print(log_var)
    '''参数集合'''
    # params = ([p for p in model.parameters()] + [log_var] + [log_vars] + [log_varss])
    # params = ([p for p in model.parameters()] + [log_var] + [log_vars])
    params = ([p for p in model.parameters()] + [log_var])
    # params = ([p for p in model.parameters()])
    # optimizer = optim.Adam([
    #     {'params': model.vits.parameters(), 'lr': 1e-5},  # VisionTransformerEncoder 部分的学习率
    #     {'params': model.reassemble.parameters(), 'lr': 1e-4},
    #     {'params': model.fusions.parameters(), 'lr': 1e-4},
    #     {'params': model.head.parameters(), 'lr': 1e-4},
    #     {'params': [log_var], 'lr': learning_rate}  # 添加 log_var
    # ], betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    writer = SummaryWriter("./ntrain_logs")
    # 训练模型
    for epoch in range(num_epochs):
        best_rmse=999
        print('epoch ', epoch + 1, 'training...')
        model.train()
        print_count = 0
        running_loss = 0.0
        train_loss = list()
        var = list()
        with alive_bar(len(train_loader), force_tty=True) as bar:
            for tobtrain in train_loader:
                images1, images2, targets, cbra = tobtrain
                images1 = images1.to(device)
                images2 = images2.to(device)
                targets = targets.to(device)
                cbra = cbra.to(device)
                '''dpt'''
                # images = torch.cat((images1,images2),dim=1)
                # outputs = model(images)
                # loss_mse = f.mse_loss(outputs, targets, reduction='mean').cpu().detach().numpy()
                # targetsbi = torch.where(cbra > 0, torch.ones_like(cbra), torch.zeros_like(cbra))
                # loss = Loss_mse_bce1(outputs, [targets, targetsbi], log_var.to(device))
                '''2 losses'''
                # images = torch.cat((images1,images2),dim=1)
                # outputs = model(images2)
                # loss_mse = f.mse_loss(outputs, targets, reduction='mean').cpu().detach().numpy()
                # loss = Loss_mse_cs(outputs, targets, log_var.to(device))
                '''another 2 losses'''
                # outputs = model(images1, images2)
                # loss_mse = f.mse_loss(outputs, targets, reduction='mean').cpu().detach().numpy()
                # targetsbi = torch.where(cbra > 0, torch.ones_like(cbra), torch.zeros_like(cbra))
                # loss = Loss_mse_dicebce(outputs, [targets, targetsbi], log_var.to(device))
                '''2 losses'''
                ypred2, ypred3 = model(images1, images2)
                targetsbi = torch.where(cbra > 0, torch.ones_like(cbra), torch.zeros_like(cbra))
                # targetsbi = targetsbi.long().view(-1).to(device)
                # ypred2 = ypred2.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 2)
                # mask_bh_gt3 = torch.where(targets > 3, torch.ones_like(targets), torch.zeros_like(targets))
                # pred_bh_mask = torch.mul(mask_bh_gt3, targets)
                # loss_mse = f.mse_loss(ypred3, targets, reduction='mean').cpu().detach().numpy()
                #
                # loss = Loss_mse_bce([ypred2, ypred3],
                #                     [targets, targetsbi],
                #                     log_var.to(device))
                combined_condition = targets > 20
                targetsbi[combined_condition] = 2
                targetsbi = targetsbi.squeeze(1).long()
                loss_mse = f.mse_loss(ypred3, targets, reduction='mean').cpu().detach().numpy()
                loss = Loss_mse_bce3([ypred3, ypred2],
                                     [targets, targetsbi],
                                     log_var.to(device))
                '''resuper'''
                # ypred1, ypred2 = model(images2)
                # targetsbi = torch.where(targets > 0, torch.ones_like(targets), torch.zeros_like(targets))
                # # combined_condition = (targetsbi == 1) & (targets > 20)
                # # targetsbi[combined_condition] = 2
                # # targetsbi = targetsbi.squeeze(1).long()
                # loss_mse = f.mse_loss(ypred1, targets, reduction='mean').cpu().detach().numpy()
                # loss = Loss_mse_dicebce([ypred1, ypred2],
                #                      [targets, targetsbi], log_var.to(device))

                optimizer.zero_grad()
                loss.backward()
                max_grad = 0
                min_grad = float('inf')
                for param in model.parameters():
                    if param.grad is not None:
                        max_grad = max(max_grad, param.grad.abs().max().item())
                        min_grad = min(min_grad, param.grad.abs().min().item())
                optimizer.step()

                train_loss.append(loss.cpu().detach().numpy())
                running_loss += loss_mse.item() * images2.size(0)

                var.append(log_var.cpu().detach().numpy())

                '''输出'''
                if print_count % 2 == 0:
                    bar.text('mse %.4f, min grad %.4f, max grad %.4f'
                             % (loss_mse.item(), min_grad, max_grad))
                print_count += 1
                bar()

        epoch_loss = np.sqrt(running_loss / len(train_loader.dataset))
        vals,names = test_epoch(model, test_loader, device, epoch)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        writer.add_scalar('train loss',
                          (np.mean(train_loss)),  # average
                          epoch)
        writer.add_scalar('train rmse',
                          (np.mean(epoch_loss)),  # average
                          epoch)
        writer.add_scalar(names[0],
                          (np.mean(vals[0])),  # average
                          epoch)
        writer.add_scalar(names[1],
                          (np.mean(vals[1])),  # average
                          epoch)
        for h in range(3):
            writer.add_scalar(names[h+2],
                              (vals[h+2]),  # average
                              epoch)
        writer.add_scalar(names[5],
                          (np.mean(vals[5])),  # average
                          epoch)
        # writer.add_scalar('weight a',
        #                   (np.mean(var)),  # average
        #                   epoch)
        scheduler.step()
        if vals[0]<best_rmse:
            model_name = "deeplabv3.pth"
            save_path = os.path.join(save_dir, model_name)
            torch.save(model.state_dict(), save_path)
        torch.save(model.state_dict(), os.path.join(save_dir, "latest_model.pth"))
    # 保存模型
    # torch.save(model.state_dict(), "unet_model.pth")

