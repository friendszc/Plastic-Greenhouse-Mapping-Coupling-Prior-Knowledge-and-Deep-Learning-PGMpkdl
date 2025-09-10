import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import glob
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import rasterio
from models.UNet import UNet
from utils.dataloader import get_datasets
from utils.metrics import get_metrics
from train import train_model, evaluate
from config import config
import warnings
from predict import predict_from_inmodel

warnings.filterwarnings("ignore")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PyTorch UNet训练脚本")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'eval'],
                        help="运行模式: train(训练) | predict(预测) | eval(评估)")
    parser.add_argument('--model', type=str, default='UNet', choices=['UNet', 'SegNet', 'DeepLab'],
                        help="模型架构: unet | segnet | deeplab")
    parser.add_argument('--regions', type=str, default=None,
                        help="覆盖config中的训练区域代码，例如 'ABDF' 表示使用A/B/D/F区域")
    parser.add_argument('--epochs', type=int, default=None,
                        help="覆盖config中的训练轮数")
    parser.add_argument('--resume', type=str, default=None,
                        help="恢复训练的检查点路径")
    parser.add_argument('--batch-size', type=int, default=None,
                        help="覆盖config中的batch size")
    parser.add_argument('--inmodel', type=str, default=None,
                        help="用于预测的模型路径")
    return parser.parse_args()


def setup_environment(args):
    """配置环境和路径"""
    os.makedirs(f"../../train/checkpoints/{args.model}", exist_ok=True)
    os.makedirs(f"../../train/logs/{args.model}", exist_ok=True)

    # 覆盖配置参数（如果命令行指定）
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.regions:
        config.regions = args.regions

    # 设置随机种子（保证可复现性）
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def init_model(args):
    """初始化模型"""
    model_map = {
        'UNet': UNet,
        # 'segnet': SegNet,
        # 'deeplab': DeepLabV3
    }
    model = model_map[args.model](n_channels=config.in_channels,
                                  n_classes=config.out_channels)

    # 恢复训练
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume))
        print(f"从 {args.resume} 恢复模型权重")

    return model.to(config.device)


def predict_mode(args, model):
    print("开始执行预测...")

    # 默认模型路径
    model_dir = os.path.dirname(args.resume) if args.resume else f"../../train/checkpoints/{args.model}"
    input_dir = os.path.join(model_dir, 'inference_inputs')
    output_dir = os.path.join(model_dir, 'prediction')
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有tif文件
    tif_files = glob.glob(os.path.join(input_dir, '*.tif'))
    if not tif_files:
        print(f"在 {input_dir} 下未找到任何.tif文件")
        return

    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        for tif_path in tqdm(tif_files, desc="预测中"):
            with rasterio.open(tif_path) as src:
                image = src.read()  # shape: (channels, height, width)
                meta = src.meta

            # 预处理
            image = image.astype(np.float32) / 255.0
            input_tensor = torch.from_numpy(image).unsqueeze(0).to(config.device)

            if input_tensor.shape[1] != config.in_channels:
                print(f"警告：图像 {tif_path} 的通道数不匹配，跳过")
                continue

            # 模型预测
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # 保存预测
            output_path = os.path.join(output_dir, os.path.basename(tif_path))
            meta.update({'count': 1, 'dtype': 'uint8'})
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(pred, 1)

    print(f"预测完成，结果保存在 {output_dir}")


def main():
    args = parse_args()
    setup_environment(args)

    # 初始化模型
    model = init_model(args)

    # summary(model, input_size=(3, 128, 128))

    # 数据加载
    train_set, val_set = get_datasets(regions=list(config.regions))
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers, pin_memory=True)

    if args.mode == 'train':
        print(f"开始训练 {args.model.upper()}，使用区域: {args.regions}, 批次大小: {config.batch_size}")
        print(f"train_set: {len(train_set)}, val_set: {len(val_set)}")

        # 训练模型
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.epochs,
            device=config.device
        )

        # 保存结果
        lw_str = f"{int(config.loss_weight.item() * 10):02d}"
        model_dir = f"../../train/{args.model}"
        os.makedirs(model_dir, exist_ok=True)
        model_fn = os.path.join(model_dir, f"{args.model}_region{''.join(config.regions)}_lossweight{lw_str}.pth")
        torch.save(model.state_dict(), model_fn)
        log_fn = os.path.join(model_dir, f"{args.model}_region{''.join(config.regions)}_lossweight{lw_str}_logs.csv")
        pd.DataFrame(history).to_csv(log_fn, index=False)
        print("训练完成！模型已保存到 checkpoints/")

    elif args.mode == 'eval':
        print("开始评估模型...")
        metrics = evaluate(model, val_loader, config.device)
        print("\n评估结果:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    elif args.mode == 'predict':
        inmodel = args.inmodel
        predict_from_inmodel(inmodel)


if __name__ == '__main__':
    main()