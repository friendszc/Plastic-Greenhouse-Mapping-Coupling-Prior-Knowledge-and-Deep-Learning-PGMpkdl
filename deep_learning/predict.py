import os
import glob
from pathlib import Path

import torch
import rasterio
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from config import config
from models.UNet import UNet
# from models.SegNet import SegNet  # 如你有其他模型可以继续引入
# from models.DeepLab import DeepLabV3

def predict_from_inmodel(inmodel: str):
    """
    从模型权重路径 inmodel 加载模型（根据路径自动识别模型名），
    对 config.test_path 中所有 TIF 文件进行预测。
    预测结果保存至 inmodel 同级目录 prediction 文件夹中。
    """

    # 自动识别模型名
    model_name = os.path.basename(inmodel).split('_')[0]

    # 初始化模型结构
    model_map = {
        'UNet': UNet,
        # 'SegNet': SegNet,
        # 'DeepLab': DeepLabV3
    }

    if model_name not in model_map:
        raise ValueError(f"[!] 未支持的模型类型: {model_name}")

    model = model_map[model_name](n_channels=config.in_channels, n_classes=config.out_channels)

    # 加载模型权重
    if not os.path.exists(inmodel):
        raise FileNotFoundError(f"[!] 模型文件不存在: {inmodel}")
    model.load_state_dict(torch.load(inmodel, map_location=config.device))
    model.to(config.device)
    model.eval()
    print(f"[✓] 成功加载模型: {model_name} 权重自 {inmodel}")

    # 输出目录
    output_dir = os.path.join(os.path.dirname(inmodel), Path(inmodel).stem+'_prediction')
    os.makedirs(output_dir, exist_ok=True)

    # 获取测试图像
    test_path = os.path.join(config.src_path, 'test_set')
    tif_files = glob.glob(os.path.join(test_path, '*.tif'))
    if not tif_files:
        print(f"[!] 未找到测试图像于: {config.test_path}")
        return

    print(f"[✓] 开始预测 {len(tif_files)} 张图像")

    with torch.no_grad():
        for tif_path in tqdm(tif_files, desc="推理中"):
            with rasterio.open(tif_path) as src:
                image = src.read()
                meta = src.meta

            image = image.astype(np.float32) / 10000.0
            input_tensor = torch.from_numpy(image).unsqueeze(0).to(config.device)

            if input_tensor.shape[1] != config.in_channels:
                print(f"⚠️ 跳过 {tif_path}，通道数为 {input_tensor.shape[1]}，应为 {config.in_channels}")
                continue

            output = model(input_tensor)
            pred = (output > 0.5).squeeze().cpu().numpy().astype(np.uint8)

            meta.update({'count': 1, 'dtype': 'uint8'})
            out_path = os.path.join(output_dir, os.path.basename(tif_path))
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(pred, 1)

    print(f"[✓] 所有预测完成，结果保存在: {output_dir}")


if __name__ == "__main__":
    inmodel = r"G:\train\UNet\UNet_regionABCD_lossweight30.pth"
    predict_from_inmodel(inmodel)