"""
intif to chips
"""

import glob
import os
import shutil
import sys
# import cv2
import numpy as np
from osgeo import gdal
import csv

def replace_Nan(arr, nodata):
    arr[np.isnan(arr)] = nodata


def createRaster(output, arr, h, w, trans=False, proj=False, nodataValue=0):
    # 波段数
    band_num = 1 if len(arr.shape) == 2 else arr.shape[0]

    # 创建栅格
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    oDS = driver.Create(output, w, h, band_num, gdal.GDT_Int16)  # 新建数据集
    if proj:
        oDS.SetProjection(proj)
    if trans:
        oDS.SetGeoTransform(trans)
    
    if band_num == 1:
        oDS.GetRasterBand(1).WriteArray(arr)
    else:
        for i in range(band_num):
            band = oDS.GetRasterBand(i + 1)
            band.WriteArray(arr[i])

    oDS.FlushCache()  # 最终将数据写入硬盘
    oDS = None  # 注意必须关闭tif文件


def tif_clip(tif, outPath, region, cliph=128, clipw=128, step=64, nodata=0, replaceNan=True):
    """
    将一张tif裁切为符合input.shape的多张tif，输出路径为在原路径下新建src和label文件夹，文件命名为index.tif
    注意：最后N行或N列像素数不足以裁切时，不进行裁切
    :param tif: 要裁切的大图
    :param dir: ‘src’ or 'label'
    :param cliph: window height, 默认128
    :param clipw: window width, 默认128
    :param nodata:
    :param replaceNan:
    :return:
    """

    # print(outPath)
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    # else:
    #     shutil.rmtree(outPath)
    #     os.makedirs(outPath)

    src = gdal.Open(tif)
    arr = src.ReadAsArray()
    # print('输入tif形状为：', arr.shape)
    if replaceNan:
        replace_Nan(arr, nodata)

    Nh = (arr.shape[-2]-cliph) // step + 1
    Nw = (arr.shape[-1]-clipw) // step + 1
    # print('Nh=%d, Nw=%d'%(Nh, Nw))

    global index

    w = 0
    h = 0
    for i in range(Nh):
        w = 0
        # print(index)
        for j in range(Nw):
            # if h <= 500 or w+win_w >= 3000:
            #     continue
            if len(arr.shape) == 2:
                window_arr = arr[h:h+win_h, w:w+win_w]
            else:
                window_arr = arr[:, h:h+win_h, w:w+win_w]
            # 输出tif
            index += 1
            output = os.path.join(outPath, f'{region}_{index}.tif')
            createRaster(output, window_arr, clipw, cliph)
            w += step
        h += step
    print("完成裁切%s, index = %d"%(tif, index))
    return index


def remove_exsisted(inPath, exts):
    tifs = []
    for ext in exts:
        tifs = glob.glob(os.path.join(inPath, "*%s.tif"%ext))
        print(len(tifs), "files removing")
        for tif in tifs:
            os.remove(tif)

def move_files_based_on_csv(input_path, csv_file, out_path):
    """
    根据 CSV 中的文件名列表，将文件从输入路径移动到输出路径

    参数:
        input_path (str): 原始文件所在的目录路径
        csv_file (str): 包含文件名的 CSV 文件路径（单列，列名为"Filename"）
        out_path (str): 目标输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(out_path, exist_ok=True)

    # 读取CSV文件中的文件名
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        filenames = [row['Filename'] for row in reader]

    # 移动文件
    for filename in filenames:
        src = os.path.join(input_path, filename)
        dst = os.path.join(out_path, filename)

        try:
            shutil.move(src, dst)
            print(f"移动成功: {dst}")
        except FileNotFoundError:
            print(f"文件不存在: {filename}")
        except Exception as e:
            print(f"移动失败 {filename}: {str(e)}")

def main():

    global index
    regions = ["A", "B", "D", "E", "F"]
    dataPath = r"G:\Data"
    sub_dirs = [
                'ds_label',
                # 'pkb_label',
                # 'RGB'
                ]

    for sub_dir in sub_dirs:
        inPath = os.path.join(dataPath, sub_dir)
        outPath = inPath+'_chips'
        testPath = os.path.join(outPath, "test_set")
        os.makedirs(testPath, exist_ok=True)
        for region in regions:
            index = 0
            intif = glob.glob(os.path.join(inPath, f'{region}_*.tif'))[0]
            tif_clip(intif, outPath, region, step=step)
        move_files_based_on_csv(outPath, csv_path, testPath)


if __name__ == '__main__':

    win_h = 128
    win_w = 128
    step = 128
    nodata = -32768
    csv_path = r"G:\Data\testSet.csv"

    main()