import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
sys.path.append('..')
from config import config
import glob
import rasterio


class CustomDataset(Dataset):
    def __init__(self, src_dir, label_dir, regions, transform=None):
        """
        Args:
            src_dir: 源图像目录
            label_dir: 标签图像目录
            regions: 包含的区域列表 (如 ['A', 'B', 'C'])
            transform: 数据增强变换
        """
        self.src_dir = src_dir
        self.label_dir = label_dir
        self.transform = transform

        # 收集符合区域条件的文件路径
        self.file_list = []
        for region in regions:
            region_files = glob.glob(os.path.join(label_dir, f'{region}_*.tif'))
            self.file_list.extend(region_files)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        label_path = self.file_list[idx]
        base_name = os.path.basename(label_path)
        src_path = os.path.join(self.src_dir, base_name)

        # 加载图像 - 使用PIL或直接numpy加载
        src_img = rasterio.open(src_path).read().astype(np.float32) / 10000.0  # 归一化
        label_img = np.array(Image.open(label_path), dtype=np.float32)

        # 转换为PyTorch需要的格式 [C, H, W]
        label_img = torch.from_numpy(label_img).unsqueeze(0)  # 添加通道维度

        # 应用数据增强
        if self.transform:
            src_img, label_img = self.transform(src_img, label_img)

        return src_img, label_img


def get_datasets(regions, val_ratio=0.3):
    """划分训练集和验证集"""
    full_dataset = CustomDataset(
        src_dir=config.src_path,
        label_dir=config.label_path,
        regions=regions
    )

    # 划分数据集
    val_size = int(val_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    return train_set, val_set