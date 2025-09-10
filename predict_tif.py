import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from pathlib import Path
from DeepLearning.models.UNet import UNet
from DeepLearning.config import config

def predict_large_image(inmodel, intif, out_tif='pred.tif',
                        window_size=128, device='cuda' if torch.cuda.is_available() else 'cpu'):

    model = UNet(n_channels=config.in_channels, n_classes=config.out_channels)
    model.load_state_dict(torch.load(inmodel, map_location=config.device))
    model.to(config.device)
    model.eval()

    intif = Path(intif)

    # 打开影像
    with rasterio.open(intif) as src:
        profile = src.profile
        height, width = src.height, src.width
        bands = src.count

        # 修改 profile 以适应单通道预测结果
        profile.update(count=1, dtype='uint8')

        # 初始化输出数组
        prediction = np.zeros((height, width), dtype=np.uint8)

        # 补零处理确保可以整除
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        padded_height = height + pad_h
        padded_width = width + pad_w

        # 遍历窗口进行预测
        for y in tqdm(range(0, padded_height, window_size), desc="Predicting"):
            for x in range(0, padded_width, window_size):
                h_win = min(window_size, height - y)
                w_win = min(window_size, width - x)

                # 读取窗口数据（超出边缘部分自动补0）
                window = Window(x, y, w_win, h_win)
                img = src.read(window=window, out_shape=(bands, window_size, window_size), boundless=True, fill_value=0)
                img = img.astype(np.float32) / 10000  # 归一化

                # 模型输入格式 (1, C, H, W)
                img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
                # print(img_tensor.shape)

                # 推理
                with torch.no_grad():
                    out = model(img_tensor)
                    # print(out.shape)
                    # pred = torch.argmax(out.squeeze(), dim=0).cpu().numpy().astype(np.uint8)
                    pred = (out > 0.5).squeeze().cpu().numpy().astype(np.uint8)
                    # print(pred.shape)
                # 裁切预测结果以匹配原始窗口尺寸
                prediction[y:y + h_win, x:x + w_win] = pred[:h_win, :w_win]

    # 写入预测结果
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(prediction, 1)

    print(f"✅ Prediction saved to {out_tif}")


inmodel = r"G:\train\UNet\UNet_regionABCD_lossweight30.pth"
for region in ['A', 'B', 'C', 'D'][1:]:
    intif = rf"G:\Data\RGB\{region}_RGB.tif"
    outtif = rf"G:\Data\preds\{region}_pred.tif"
    predict_large_image(
        inmodel=inmodel,
        intif=intif,
        out_tif=outtif
    )
