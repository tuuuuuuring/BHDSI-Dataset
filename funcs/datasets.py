import torch
from torch.utils.data import Dataset
from torch.nn import functional as f
from funcs.findhighest import ReadTif


class CustomDataset64(Dataset):
    def __init__(self, image_paths1, image_paths2, target_paths):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.target_paths1 = target_paths

    def __len__(self):
        return len(self.image_paths1)

    def __getitem__(self, idx):
        image1 = ReadTif(self.image_paths1[idx])
        image1 = torch.where(image1 < -50, -50, image1)
        image2 = ReadTif(self.image_paths2[idx])
        image2 = torch.where(image2 < 0, 0, image2)
        target = ReadTif(self.target_paths1[idx])
        target = f.avg_pool2d(target, kernel_size=4, stride=4)
        return image1, image2, target, target


class CustomDatasetGoogle(Dataset):
    def __init__(self, image_paths1, image_paths2, target_paths):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.target_paths1 = target_paths

    def __len__(self):
        return len(self.image_paths1)

    def __getitem__(self, idx):
        image1 = ReadTif(self.image_paths1[idx])
        image1 = torch.where(image1 < -50, -50, image1)
        image2 = ReadTif(self.image_paths2[idx])
        image2 = torch.where(image2 < 0, 0, image2)
        target = ReadTif(self.target_paths1[idx])
        return image1, image2, target


class CustomDataset(Dataset):
    def __init__(self, image_paths1, image_paths2, target_paths1, target_paths2):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.target_paths1 = target_paths1
        self.target_paths2 = target_paths2

    def __len__(self):
        return len(self.image_paths1)

    def __getitem__(self, idx):
        image1 = ReadTif(self.image_paths1[idx])
        image1 = torch.where(image1 < -50, -50, image1)
        image2 = ReadTif(self.image_paths2[idx])
        image2 = torch.where(image2 < 0, 0, image2)
        target1 = ReadTif(self.target_paths1[idx])
        # img_bh = target1/3
        #
        # # 定义 img5, img4, img3
        # img5 = img_bh * 5
        # img4 = img_bh * 3.1  # 原来是3.1
        # img3 = img_bh * 3
        #
        # # 处理 >= 30 的元素
        # mask_30 = img_bh >= 30
        # img_bh[mask_30] = img5[mask_30]
        #
        # # 处理 >= 20 且 < 30 的元素
        # mask_20_30 = torch.logical_and(img_bh >= 20, img_bh < 30)
        # img_bh[mask_20_30] = img4[mask_20_30]
        #
        # # 处理 < 20 的元素
        # mask_20 = img_bh < 20
        # img_bh[mask_20] = img3[mask_20]
        #
        # # 处理 < 3 的元素
        # img_bh[img_bh < 3] = 0
        #
        # # 如果需要，将处理后的 img_bh 赋值回 target1
        # target1 = img_bh
        target2 = ReadTif(self.target_paths2[idx])
        return image1, image2, target1, target2