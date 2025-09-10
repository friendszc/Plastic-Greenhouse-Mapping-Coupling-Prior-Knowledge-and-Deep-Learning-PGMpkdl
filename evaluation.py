import numpy as np
from pathlib import Path
from DeepLearning.config import config
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import rasterio
import csv


def calculate_metrics(pred, label):
    """计算各种评价指标"""
    cm = confusion_matrix(label.flatten(), pred.flatten())
    tn, fp, fn, tp = cm.ravel()

    # 计算各项指标
    OA = (tp + tn) / (tp + tn + fp + fn)  # 总体精度
    PA = tp / (tp + fn)  # 生产者精度/召回率
    UA = tp / (tp + fp)  # 用户精度/精确率
    F1 = 2 * (PA * UA) / (PA + UA)  # F1分数
    kappa = cohen_kappa_score(label.flatten(), pred.flatten())
    IOU = tp / (tp + fp + fn)  # 交并比

    return {
        'OA': OA,
        'PA': PA,
        'UA': UA,
        'F1': F1,
        'kappa': kappa,
        'IOU': IOU,
        'confusion_matrix': cm
    }


def process_regions(predPath, labelPath):
    """处理各个区域并计算指标"""
    pred_path = Path(predPath)
    label_path = Path(labelPath)

    results = {}

    # 处理每个区域
    preds_all = []
    labels_all = []
    for region in regions:
        preds_region = []
        labels_region = []

        # 查找对应区域的文件
        pred_files = list(pred_path.glob(f'{region}_*.tif'))
        label_files = list(label_path.glob(f'{region}_*.tif'))
        print(f'Region {region} 匹配到pred files {len(pred_files)} 个， label_files {len(label_files)} 个')

        for pred_file, label_file in zip(pred_files, label_files):
            # 使用rasterio读取TIFF文件
            with rasterio.open(pred_file) as src:
                pred = src.read(1)  # 读取第一个波段
            with rasterio.open(label_file) as src:
                label = src.read(1)  # 读取第一个波段

            # 确保二值图像(0和1)
            pred = (pred > 0).astype(np.uint8)
            label = (label > 0).astype(np.uint8)

            preds_all.extend(pred.flatten())
            labels_all.extend(label.flatten())
            preds_region.extend(pred.flatten())
            labels_region.extend(label.flatten())

        # print(np.unique(preds_region))
        # print(np.unique(labels_region))
        # exit(0)

        # 计算指标
        results[region] = calculate_metrics(np.array(preds_region), np.array(labels_region))

    # 计算整体指标(合并所有区域)

    results['AllRegions'] = calculate_metrics(np.array(preds_all), np.array(labels_all))

    return results


def print_results(results):
    """打印结果"""
    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        'Region', 'OA', 'PA', 'UA', 'F1', 'Kappa', 'IOU'))
    print("-" * 80)

    for region, metrics in results.items():
        print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            region,
            metrics['OA'],
            metrics['PA'],
            metrics['UA'],
            metrics['F1'],
            metrics['kappa'],
            metrics['IOU']
        ))


def save_results_to_csv(results_dict, output_path):
    """
    保存多个来源的评估结果到一个CSV文件。
    :param results_dict: dict，键是来源名（如 'DL'），值是每个来源的结果dict（region -> metrics）
    :param output_path: 保存路径
    """
    all_rows = []
    for source_name, result in results_dict.items():
        for region, metrics in result.items():
            row = {
                'source': source_name,
                'Region': region,
                'OA': round(metrics['OA'], 4),
                'PA': round(metrics['PA'], 4),
                'UA': round(metrics['UA'], 4),
                'F1': round(metrics['F1'], 4),
                'Kappa': round(metrics['kappa'], 4),
                'IOU': round(metrics['IOU'], 4)
            }
            all_rows.append(row)

    # 写入CSV
    fieldnames = ['source', 'Region', 'OA', 'PA', 'UA', 'F1', 'Kappa', 'IOU']
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    # 使用示例
    pkbPath = r"G:\Data\pkb_label_chips\test_set"
    predPath = r"G:\train\UNet\UNet_regionABCD_lossweight30_prediction"
    labelPath = r"G:\Data\ds_label_chips\test_set"
    regions = config.regions

    results_pkb_DL = process_regions(predPath, labelPath)
    results_pkb = process_regions(pkbPath, labelPath)

    print('*********** pkb_DL_results ***********')
    print_results(results_pkb_DL)
    print('************ pkb_results *************')
    print_results(results_pkb)


    # 保存结果到CSV
    output_csv = Path(predPath).with_suffix('.csv')
    save_results_to_csv({'pkb_DL': results_pkb_DL, 'pkb': results_pkb}, output_csv)
    print(f"\n结果已保存到: {output_csv}")