import torch
from torch.masked import masked_tensor


def calculate_iou(pred, target, threshold=0.5):
    # 假设pred和target都是形状为[B, C, H, W]的tensor，其中B是batch size，C是类别数（但在这个例子中我们只需要一个类别），H和W是高度和宽度
    # 如果你的tensor是二维的（例如[H, W]），则可以直接使用下面的代码

    # 将预测和目标二值化（如果它们还没有被二值化）
    pred = (pred > threshold).float()  # 如果你有阈值需要应用
    target = (target > threshold).float()  # 假设target已经是二值化的

    # 计算交集：两个tensor的逐元素乘法后求和
    intersection = (pred * target).sum()

    # 计算并集：两个tensor的逐元素加法后求和，然后减去交集（因为交集被加了两次）
    union = (pred + target).sum() - intersection

    iou = intersection / union

    return iou.item()


def relative_absolute_error(predicted, target,mask):
    predicted = predicted.float()
    target = target.float()

    absolute_diff=torch.zeros_like(target)
    relative_error=torch.zeros_like(target)
    # 计算绝对差值
    absolute_diff[mask] = torch.abs(predicted[mask] - target[mask])
    # 计算相对绝对误差
    relative_error[mask] = absolute_diff[mask] / torch.abs(target[mask])
    # 计算平均相对绝对误差
    rae = torch.mean(relative_error[mask])
    return rae


def depth_accuracy(predicted, target,mask, threshold=1.25 ):
    """
    计算单目深度估计的accuracy with threshold。

    :param predicted: 模型的预测深度值张量
    :param target: 真实深度值张量
    :param threshold: 用于计算准确度的阈值，默认为1.25
    :param eps: 一个非常小的值，用于避免除以零，默认为1e-6
    :return: 三个准确度值，分别表示δ < threshold, δ < threshold², δ < threshold³
    """
    # 确保输入的张量是浮点类型
    predicted = predicted.float()
    target = target.float()

    # 计算正向和反向比值
    mpre = masked_tensor(predicted,mask)
    mtar = masked_tensor(target,mask)
    ratio = torch.max(mpre / mtar, mtar / mpre)

    # 计算不同阈值下的准确度
    maxvs=(ratio>0).sum().item()
    accuracy_1 = (ratio < threshold).sum().item()/maxvs
    accuracy_2 = (ratio < threshold ** 2).sum().item()/maxvs
    accuracy_3 = (ratio < threshold ** 3).sum().item()/maxvs

    return accuracy_1, accuracy_2, accuracy_3


def r2_score(predicted, target):
    """
    计算两个张量之间的决定系数 R^2。

    :param predicted: 模型的预测值张量
    :param target: 真实值张量
    :return: 决定系数 R^2
    """
    # 确保输入的张量是浮点类型
    predicted = predicted.float()
    target = target.float()

    # 计算总平方和（total sum of squares）
    target_mean = target.mean()
    ss_total = torch.sum((target - target_mean) ** 2)

    # 计算残差平方和（residual sum of squares）
    ss_residual = torch.sum((target - predicted) ** 2)

    # 计算 R^2
    r2 = 1 - ss_residual / ss_total
    return r2
