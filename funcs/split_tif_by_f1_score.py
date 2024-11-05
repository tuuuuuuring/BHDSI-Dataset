import numpy as np
from osgeo import gdal
import rasterio
import os
import numpy as np
from osgeo import gdal

def read_tif(file_path):
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    return array, geotransform, projection

def write_tif(array, geotransform, projection, output_path):
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(output_path, array.shape[1], array.shape[0], 1, gdal.GDT_Byte)
    out_raster.SetGeoTransform(geotransform)
    out_raster.SetProjection(projection)
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.FlushCache()

def check_two_images(image1_path, image2_path, output_image_path):
    # 读取两张影像
    image1, geotransform, projection = read_tif(image1_path)
    image2, _, _ = read_tif(image2_path)

    # 确保两张影像大小相同
    assert image1.shape == image2.shape, f"两张影像大小不一致,{image1.shape}!= {image2.shape}"

    # 生成新的影像
    new_image = np.zeros_like(image1, dtype=np.uint8)

    # 逐像素比较
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if image1[i, j] > 1 and image2[i, j] > 1:
                new_image[i, j] = 1
            else:
                new_image[i, j] = 0

    # 写入新的影像
    output_path = output_image_path
    write_tif(new_image, geotransform, projection, output_path)

    print(f'New image saved to {output_path}')


def is_overlapping(x, y, block_size, used_blocks):
    for (ux, uy) in used_blocks:
        if not (x + block_size[0] <= ux or x >= ux + block_size[0] or y + block_size[1] <= uy or y >= uy + block_size[1]):
            return True
    return False


def is_point_in_rectangle(rect, point):
    x, y = point
    a, b = rect
    # 判断点是否在矩形内
    return abs(x - a) < 256 and abs(y - b) < 256

def is_point_in_any_rectangle(rectangles, point):
    if rectangles == []:
        return False
    for rect in rectangles:
        if abs(point[0] - rect[0]) < 256 and abs(point[1] - rect[1]) < 256:
            return True
    return False


def write_four_tif(input_tif_folder,i, j, transform, output_tif_folder):
    for name in ['_bhrf.tif', 'cbra2018.tif', 'sentinel1_AREA.tif','sentinel2_AREA.tif']:
        input_tif = os.path.join(input_tif_folder, name)
        print(input_tif)
        array, geotransform, projection = read_tif(input_tif)
        block = array[i:i + 256, j:j + 256]
        name = name.replace('.tif', '')
        output_tif_path = os.path.join(output_tif_folder, f'{name}_{i}_{j}.tif')
        print(output_tif_path)
        write_tif(array=block, geotransform=transform, projection=projection, output_path=output_tif_path)



def crop_image(image_dir, image_path, block_size, step_size,output_tif_dir):
    # output_tif_dir = r'D:\gis pro\0821\2-beijing'

    rects = []
    image, geotransform, projection = read_tif(image_path)
    i, j, count = 0, 0, 0
    # print(count)
    while i < image.shape[0] - block_size[0]:
        j = 0
        while j < image.shape[1] - block_size[1]:
            if not is_point_in_any_rectangle(rects, [i, j]):
                block = image[i:i + block_size[0], j:j + block_size[1]]
                if np.sum(block == 1) > 13963:
                    rects.append([i,j])
                    # output_tif_path = os.path.join(output_tif_dir,f'output_{i}_{j}.tif')
                    transform = list(geotransform)
                    print(transform)
                    transform[0] = geotransform[0] + geotransform[1] * j + i * geotransform[2]
                    transform[3] = geotransform[3] + geotransform[4] * j + i * geotransform[5]
                    # write_tif(array=block, geotransform=transform, projection=projection, output_path=output_tif_path)
                    print('输入', image_dir)
                    write_four_tif(image_dir, i, j, transform,output_tif_dir)
                    j += block_size[1]
                    count += 1
                    print(count, rects)
                else:
                    j += 1
            else:
                j += 1
        i += 1
    return rects


def calculate_f1_score(image_path):
    dataset = gdal.Open(image_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()

    TP = np.sum(array == 1)
    FP = np.sum(array == 0)
    FN = 0  # 假设没有假阴性
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(image_path, 'f1 分数：', f1_score)
    return f1_score

def calculate_f1_score_block(array):

    # 此处TP值只需要大于13963即可满足F1大于0.35
    TP = np.sum(array == 1)
    FP = np.sum(array == 0)
    FN = 0  # 假设没有假阴性
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def test():
    # image1_path = r'D:\gis pro\0821\2-beijing\_bhrf.tif'
    # image2_path = r'D:\gis pro\0821\2-beijing\cbra2018.tif'
    # output_image_path = r'D:\gis pro\0821\2-beijing\output.tif'
    # check_two_images(image1_path,image2_path,output_image_path)
    # for file in os.listdir(r'D:\gis pro\0821\2-beijing'):
    #     if file.endswith('.tif'):
    #         file_name = os.path.join(r'D:\gis pro\0821\2-beijing', file)
    #
    #         calculate_f1_score(file_name)

    # 读取影像
    image_path = r'D:\gis pro\0821\2-beijing\output.tif'

    # # 设置块大小和步长
    # block_size = (256, 256)
    # step_size = 1  # 初始步长为1像素
    #
    # # 裁剪影像
    # blocks = crop_image(image_path, block_size, step_size)

    # 保存裁剪后的影像块
    # output_dir = r'D:\gis pro\0821\2-beijing'
    # for idx, (block, i, j) in enumerate(blocks):
    #     output_path = f'{output_dir}/block_{i}_{j}.tif'
    #     write_tif(block, geotransform, projection, output_path)
    #
    # print(f'裁剪完成，共生成 {len(blocks)} 个块')



def traverse_and_add_info(root_folder):
    folder_info = []
    for root, dirs, files in os.walk(root_folder):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            folder_info.append(dir_path)
            # 在这里添加你需要的其他信息
            # print(f"Processing folder: {dir_path}")
    return folder_info



input_tif_dir = r'D:\ky\trdst3'
output_tif_dir = r'D:\paisenPR\bhr_by_dl\data8'
if __name__ == '__main__':
    # test()
    if not os.path.exists(output_tif_dir):
        os.makedirs(output_tif_dir)
    folder_info = traverse_and_add_info(input_tif_dir)
    for folder in folder_info:
        output_folder = folder.replace(r'D:\ky\trdst3', r'D:\paisenPR\bhr_by_dl\data8')
        # print(output_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        tif_bhrf = os.path.join(folder, '_bhrf.tif')
        tif_cbra = os.path.join(folder, 'cbra2018.tif')
        output_image_path = tif_cbra.replace('cbra2018.tif','output.tif')
        check_two_images(image1_path=tif_bhrf,image2_path=tif_cbra,output_image_path=output_image_path)
        print(output_folder)
        crop_image(image_dir= folder, image_path=output_image_path,block_size=(256, 256), step_size=1, output_tif_dir=output_folder)



        ## 读取影像
        # true_image_path = 'path_to_true_image.tif'
        # pred_image_path = 'path_to_pred_image.tif'
        # true_image, geotransform, projection = read_tif(true_image_path)
        # pred_image, _, _ = read_tif(pred_image_path)
        #
        # # 处理影像
        # block_size = (256, 256)
        # f1_threshold = 0.35
        # valid_blocks = process_images(true_image, pred_image, block_size, f1_threshold)
        #
        # # 创建新的影像
        # new_image = np.zeros_like(true_image)
        # for block, i, j in valid_blocks:
        #     new_image[i:i+block_size[0], j:j+block_size[1]] = block
        #
        # # 写入新的影像
        # output_path = 'path_to_output_image.tif'
        # write_tif(new_image, geotransform, projection, output_path)
        #
        # print(f'New image saved to {output_path}')
