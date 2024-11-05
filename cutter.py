# -*- coding: utf-8 -*-

"""
代码简介
按照所需要的size的矩形批量对多波段整幅影像进行裁剪并输出多个矩形小块影像；
非矩形用nan补全；
参数(输入栅格文件存放文件夹路径，输出多个矩形栅格文件存放文件夹路径，使用波段数量，裁剪矩形大小)
~~~~~~~~~~~~~~~~
code by kunqi
Aerospace Information Research Institute, Chinese Academy of Sciences
kouwenqi22@mails.ucas.ac.cn
"""
import glob
import concurrent.futures
import numpy as np
from osgeo import gdal
import os
import matplotlib.pyplot as plt
import sys


# 读tiff
def readTIFF(tifpath, bandnum):

    image = gdal.Open(tifpath)  # 打开影像
    if image == None:
        print(tifpath + "该tif不能打开!")
        return
    im_width = image.RasterXSize  # 栅格矩阵的列数
    im_height = image.RasterYSize  # 栅格矩阵的行数
    im_bands = image.RasterCount  # 波段数
    # im_proj = image.GetProjection()  # 获取投影信息坐标系
    # im_geotrans = image.GetGeoTransform()  # 仿射矩阵
    print(tifpath+':{}行，{}列，{}波段, 取出第{}层.'.format(im_width, im_height, im_bands, bandnum))
    if bandnum < 1 or bandnum > im_bands:
        print("指定的波段索引超出范围。")
        return None, None, None
        # 读取指定波段的数据
    im_data = image.GetRasterBand(bandnum).ReadAsArray(0, 0, im_width, im_height)
    del image  # 减少冗余
    return im_data, im_height, im_width


def find_files_with_keyword(folder_path, keyword):
    # original_path = os.getcwd()
    # 切换到指定文件夹路径
    os.chdir(folder_path)
    # 使用 glob 模块查找包含特定字段的文件
    files = glob.glob('*{}*'.format(keyword))
    # 返回文件路径列表
    # os.chdir(original_path)
    return [os.path.abspath(file) for file in files]


def is_overlapping(x1, y1, x2, y2):
    """
    判断两个 256x256 格子是否重叠

    参数:
    x1, y1: 第一个格子的左上角点坐标
    x2, y2: 第二个格子的左上角点坐标

    返回值:
    如果两个格子重叠，返回 True，否则返回 False
    """
    # 计算第一个格子的右下角点坐标
    x1_end = x1 + 255
    y1_end = y1 + 255

    # 计算第二个格子的右下角点坐标
    x2_end = x2 + 255
    y2_end = y2 + 255

    # 判断两个格子是否水平方向上有重叠
    horizontal_overlap = (x1 <= x2 <= x1_end) or (x2 <= x1 <= x2_end)

    # 判断两个格子是否垂直方向上有重叠
    vertical_overlap = (y1 <= y2 <= y1_end) or (y2 <= y1 <= y2_end)

    # 如果水平和垂直方向上都有重叠，则两个格子重叠
    return horizontal_overlap and vertical_overlap


def plot_squares(point_list):
    """
    绘制以点集作为左上角的 256×256 正方形

    参数:
    point_list: 点集，格式为 [(x1, y1), (x2, y2), ...]
    """
    # 创建图形
    fig, ax = plt.subplots()

    # 遍历点集，绘制正方形
    for point in point_list:
        x, y = point
        # 计算正方形的右下角坐标
        x_end = x + 256
        y_end = y + 256
        # 绘制正方形
        rect = plt.Rectangle((x, y), 256, 256, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # 设置坐标轴范围
    ax.set_xlim(0, 800)  # 根据实际情况调整范围
    ax.set_ylim(0, 600)  # 根据实际情况调整范围

    # 显示图形
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of Squares')
    plt.grid(True)
    plt.show()


def calculate_f1_score(predicted_data, true_data):
    # 首先，我们需要计算 TP, FP, FN
    tp = np.sum(np.logical_and(predicted_data == 1, true_data == 1))
    fp = np.sum(np.logical_and(predicted_data == 1, true_data == 0))
    fn = np.sum(np.logical_and(predicted_data == 0, true_data == 1))

    # 接下来，计算 Precision 和 Recall
    # 需要检查分母是否为零，以避免除以零的错误
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 最后计算 F1 分数，同样要检查分母是否为零
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score
# def calculate_f1_score(predicted_data, true_data):
#     # 将栅格数据转换为二分类问题的混淆矩阵
#     TP = np.sum(np.logical_and(predicted_data == 1, true_data == 1))
#     FP = np.sum(np.logical_and(predicted_data == 1, true_data == 0))
#     FN = np.sum(np.logical_and(predicted_data == 0, true_data == 1))
#
#     # 计算准确率和召回率
#     if TP == 0 and FP == 0:
#         f1_score = 0
#     else:
#         precision = TP / (TP + FP)
#         recall = TP / (TP + FN)
#         # 检查分母是否为0
#         if precision + recall == 0:
#             f1_score = 0
#         else:
#             # 计算 F1 分数
#             f1_score = 2 * precision * recall / (precision + recall)
#
#     return f1_score


def intersection_over_union(predicted_data, true_data):
    # 计算交集
    intersection = np.logical_and(predicted_data, true_data).sum()
    # 计算并集
    union = np.logical_or(predicted_data, true_data).sum()

    # 计算交并比
    iou = intersection / union if union > 0 else 0

    return iou


def noneinfo_judgement(wsf, bhrf, b):
    wsfdata, wsf_h, wsf_l = readTIFF(wsf, 1)
    bhrfdata, bhrf_h, bhrf_l = readTIFF(bhrf, 1)
    messmatrix = np.where(bhrfdata > 0, 1, 0)
    wsfmatrix = np.where(wsfdata > 0, 1, 0)
    print(messmatrix.shape)
    nofh = wsf_h - b + 1
    nofl = wsf_l - b + 1
    print(nofh, nofl)
    print('moving')
    dependence = [(99999, 99999)]
    for i in range(nofh):
        for j in range(nofl):
            overlap = False
            for point in dependence:
                x, y = point
                overlap = is_overlapping(x, y, j, i)
                if overlap:
                    break
            if overlap:
                # print('wow')
                continue
            else:
                submatrix = messmatrix[i:i + b, j:j + b]
                submatrix_wsf = wsfmatrix[i:i + b, j:j + b]
                f1 = calculate_f1_score(submatrix, submatrix_wsf)
                # iou = intersection_over_union(submatrix, submatrix_wsf)
                if f1 > 0.35:
                    dependence.append((j, i))
                    # print(f1)
    print('done')
    del dependence[0]
    print(dependence)
    return dependence


def noneinfo_judgement_nosliding(wsf, bhrf, b):
    wsfdata, wsf_h, wsf_l = readTIFF(wsf, 1)
    bhrfdata, bhrf_h, bhrf_l = readTIFF(bhrf, 1)
    messmatrix = np.where(bhrfdata > 0, 1, 0)
    wsfmatrix = np.where(wsfdata > 0, 1, 0)
    print(messmatrix.shape)

    print('griding')
    dependence = []
    for i in range(0, wsf_h-b, b):
        for j in range(0, wsf_l-b, b):
            submatrix = messmatrix[i:i + b, j:j + b]
            submatrix_wsf = wsfmatrix[i:i + b, j:j + b]
            f1 = calculate_f1_score(submatrix, submatrix_wsf)
            # iou = intersection_over_union(submatrix, submatrix_wsf)
            amount = np.sum(submatrix_wsf > 0)
            rate = amount / b * b
            if f1 > 0.35 or rate < 0.01:
                dependence.append((j, i))
    print('done')
    print(dependence)
    return dependence


def nonebd_judgement(wdf,b):
    wsfdata, wsf_h, wsf_l = readTIFF(wdf, 1)
    wsfmetrix = np.where(wsfdata > 0, 1, 0)
    print(wsfmetrix.shape)
    nofh = wsf_h - b + 1
    nofl = wsf_l - b + 1
    print(nofh, nofl)
    print('moving')
    dependence = [(99999, 99999)]
    for i in range(nofh):
        for j in range(nofl):
            overlap = False
            for point in dependence:
                x, y = point
                overlap = is_overlapping(x, y, j, i)
                if overlap:
                    break
            if overlap:
                # print('wow')
                continue
            else:
                submatrix_wsf = wsfmetrix[i:i + b, j:j + b]
                amount=np.sum(submatrix_wsf>0)
                rate=amount/b*b
                if rate<0.01:
                    dependence.append((j,i))
    print('done')
    del dependence[0]
    print(dependence)
    return dependence


# 裁剪(输入栅格文件存放文件夹路径，输出栅格文件存放文件夹路径，使用波段数量，裁剪大小)
def clipp(int_path, out_path, use_band, size, pixcont):
    inList = [name for name in os.listdir(int_path)]
    use_bandc = use_band
    for file in inList:
        use_band = use_bandc
        if file.endswith('.tif'):
            print("待裁剪影像：", file)
            input_raster = os.path.join(int_path, file)
            # 输出文件的完整路径
            output_raster = os.path.join(out_path, file.strip(".tif"))
            if not os.path.exists(output_raster):
                os.makedirs(output_raster)
            in_ds = gdal.Open(input_raster)  # 读取要切的原图
            if in_ds is None:
                print("打开失败！")
            else:
                print("打开成功！")
                width = in_ds.RasterXSize  # 获取数据宽度
                height = in_ds.RasterYSize  # 获取数据高度
                outbandsize = in_ds.RasterCount  # 获取数据波段数
                im_geotrans = in_ds.GetGeoTransform()  # 获取仿射矩阵信息
                im_proj = in_ds.GetProjection()  # 获取投影信息
                datatype = in_ds.GetRasterBand(1).DataType
                print("==================", file, "影像信息======================")
                print("width:", width, "height:", height, "outbandsize:", outbandsize)
                if use_band == 0:
                    use_band = outbandsize
                if outbandsize < use_band:
                    print("影像波段数小于所使用波段数！")
                else:
                    # 读取原图中所需的前ues_band个波段，并将前use_band个波段读入“data_all”
                    int_data_all = []
                    for inband in range(use_band):
                        int_band = in_ds.GetRasterBand(inband + 1)
                        int_data_all.append(int_band)
                    print('总共打开，并读取到内存的影像波段数目：{0}'.format(len(int_data_all)))

                    # 定义切图的大小（矩形框）
                    size = size
                    if size > width or size > height:
                        print("裁剪尺寸大于原始影像，请重新确定输入！")
                    else:
                        # 定义切图的起始点坐标
                        col_num = int(width / size)  # 宽度可以分成几块
                        row_num = int(height / size)  # 高度可以分成几块
                        # if (width % size != 0):
                        #     col_num += 1
                        # if (height % size != 0):
                        #     row_num += 1
                        num = 1  # 记录一共有多少块
                        print("row_num:%d   col_num:%d" % (row_num, col_num))
                        for point in pixcont:
                            offset_y, offset_x = point

                            # for i in range(col_num):  # 0-2
                            #     for j in range(row_num):  # 0-4
                            #         if pixcont[j][i] > 0:
                            #             offset_x = j * size
                            #             offset_y = i * size
                            # 从每个波段中切需要的矩形框内的数据
                            b_ysize = min(width - offset_y, size)
                            b_xsize = min(height - offset_x, size)
                            print(
                                "width:%d     height:%d    offset_x:%d    offset_y:%d     b_xsize:%d     b_ysize:%d" % (
                                    width, height, offset_x, offset_y, b_xsize, b_ysize))
                            # print("\n")
                            out_data_all = []
                            for band in range(use_band):
                                out_data_band = int_data_all[band].ReadAsArray(offset_y, offset_x, b_ysize, b_xsize)
                                # out_data_band[np.where(out_data_band < 0)] = 0
                                out_data_band[np.isnan(out_data_band)] = 0
                                print("min", np.min(out_data_band), "max", np.max(out_data_band))
                                out_data_all.append(out_data_band)
                            # print("out_data第{0}矩形已成功写入：{1}个波段".format(num, len(out_data_all)))
                            # 获取Tif的驱动，为创建切出来的图文件做准备
                            gtif_driver = gdal.GetDriverByName("GTiff")
                            file = output_raster + '\%04d.tif' % num
                            print("out_file", file)
                            num += 1
                            # 创建切出来的要存的文件
                            out_ds = gtif_driver.Create(file, size, size, outbandsize, datatype)
                            # print("create new tif file succeed")
                            # 获取原图的原点坐标信息
                            ori_transform = in_ds.GetGeoTransform()
                            # 读取原图仿射变换参数值
                            top_left_x = ori_transform[0]  # 左上角x坐标
                            w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
                            top_left_y = ori_transform[3]  # 左上角y坐标
                            n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率

                            # 根据反射变换参数计算新图的原点坐标
                            top_left_x = top_left_x - offset_y * n_s_pixel_resolution
                            top_left_y = top_left_y - offset_x * w_e_pixel_resolution
                            print("top_left_x", top_left_x, "top_left_y", top_left_y)

                            # 将计算后的值组装为一个元组，以方便设置
                            dst_transform = (
                                top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4],
                                ori_transform[5])

                            # 设置裁剪出来图的原点坐标
                            out_ds.SetGeoTransform(dst_transform)

                            # 设置SRS属性（投影信息）
                            out_ds.SetProjection(in_ds.GetProjection())

                            # 写入目标文件
                            for w_band in range(use_band):
                                out_ds.GetRasterBand(w_band + 1).WriteArray(out_data_all[w_band])

                            # 将缓存写入磁盘
                            out_ds.FlushCache()
                            print("=============已成功写入第{}个矩形===============".format(num-1))
                            del out_ds

            del in_ds


def process_imgdir(i, imgdir_path):
    dependence = noneinfo_judgement(cbra_path[i], bhrf_path[i], rectangle_size)
    op = out_filepath + '/' + f'cut{i+1}'
    clipp(imgdir_path, op, int(band_num), int(rectangle_size), dependence)
    txt_path = os.path.join(op, 'dedependence.txt')
    with open(txt_path, 'w') as f:
        # 遍历 dependence 列表并逐行写入文件
        for item in dependence:
            # 将每个元组转换为字符串，并写入文件
            f.write(f"{item[0]}, {item[1]}\n")


if __name__ == '__main__':
    img_filepath = r"D:\ky\trdst3"
    out_filepath = "./data8"   # data 11 256*256 F1 score > 35%  data2 61 256*256 iou > 40%  data3 1  dfc2023
    # data4 62 256*256 F1 score > 35%  shhh 山河湖海
    imgdir_paths = [name for name in os.listdir(img_filepath)]

    for i, sdf in enumerate(imgdir_paths):
        imgdir_paths[i] = os.path.join(img_filepath, sdf)
        new_folder_name = f"cut{i + 1}"
        new_folder_path = os.path.join(out_filepath, new_folder_name)

        # 检查文件夹是否已存在，如果不存在则创建
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Created folder: {new_folder_path}")
        else:
            print(f"Folder {new_folder_path} already exists")

    band_num = 0  # 0 stands for all
    rectangle_size = 256
    if len(sys.argv) > 2:
        img_filepath = sys.argv[1]
        out_filepath = sys.argv[2]
        band_num = sys.argv[3]
        rectangle_size = sys.argv[4]

    cbra_path = []
    bhrf_path = []
    for ip in imgdir_paths:
        cvgd = os.getcwd()
        cbra_path.append(find_files_with_keyword(ip, '2018')[0])
        bhrf_path.append(find_files_with_keyword(ip, 'bhrf')[0])
        os.chdir(cvgd)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # 提交任务并获取 Future 对象列表
        futures = [executor.submit(process_imgdir, i, imgdir_path) for i, imgdir_path in enumerate(imgdir_paths)]
        # 等待所有任务完成
        concurrent.futures.wait(futures)

    # with alive_bar(len(imgdir_paths), force_tty=True) as bar:
    #     for i, ip in enumerate(imgdir_paths):
    #         dependence = noneinfo_judgement(cbra_path[i], bhrf_path[i], rectangle_size)
    #         op = out_filepath + '/' + f'cut{i+37}'
    #         clipp(ip, op, int(band_num), int(rectangle_size), dependence)
    #         bar()

        # 北京116，41.3  上海121，31.3
        # 深圳香港113.5，23.8  成都103.5, 31.3  重庆106，31.3  苏杭118.5，31.3  西安108.5，36.3  武汉113.5，31.3
        # 郑州113.5，36.3+（洛阳）  南京118.5，33.8  天津116，41.3  长沙111，28.8  东莞113.5，23.8  佛山111，23.8  合肥116，33.8
        # 哈尔滨126，46.3  长春123.5，46.3  沈阳121+123.5，43.8  太原111，38.8  石家庄113.5，38.8  兰州103.5，36.3
        # 济南116，38.8  南昌113.5，28.8  福州118.5，26.3  南宁106，23.8  海口108.5，21.3  昆明101，26.3  贵阳106，28.8
        # 呼和浩特111, 41.3  西宁101，38.8  珠海澳门113.5，23.8  银川106，38.8  扬州118.5，33.8  无锡118.5，33.8  拉萨91，31.3
        # 乌鲁木齐86，43.8+46.3(无了）
        # 保定113.5，38.8+41.3 常州118.5，33.8（南京） 大连121，41.3 鄂尔多斯108.5，41.3 广州111+113.5，23.8 惠州113.5，23.8（珠海）
        # 嘉兴118.5，31.3（苏杭） 金华118.5，31.3（苏杭） 洛阳111，36.3 南通118.5+121，33.8（南京） 宁波121，31.3（上海） 青岛118.5，36.3
        # 泉州116+118.5，26.3 三亚108.5，18.8 厦门116，26.3（泉州） 汕头116，23.8 绍兴118.5，31.3（苏杭） 台州121，28.8 唐山116，41.3（天津）
        # 温州118.5，28.8 芜湖116，33.8 徐州116，36.3 烟台121，38.8 中山111，23.8（广州）

