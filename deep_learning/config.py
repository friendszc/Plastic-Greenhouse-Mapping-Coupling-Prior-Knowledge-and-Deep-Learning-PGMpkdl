import os.path
from pathlib import Path

import torch


class Config:

    # 设备配置
    device = torch.device("cuda:0")

    # 数据参数
    root_path = Path(__file__).parents[1]
    data_path = root_path.joinpath("data")
    src_path = os.path.join(data_path, 'RGB_chips')
    label_path = os.path.join(data_path, 'pkb_label_chips')
    batch_size = 64
    num_workers = 0
    img_size = 128

    # 模型参数
    in_channels = 3
    out_channels = 1
    loss_weight = torch.tensor([3], device=device)

    # 训练参数
    epochs = 200
    lr = 1e-3
    patience = 10
    factor = 0.5
    early_stop = 2*patience

    # 其他
    regions = [
                "A",
                "B",
                "C",
                "D",
            ]


config = Config()