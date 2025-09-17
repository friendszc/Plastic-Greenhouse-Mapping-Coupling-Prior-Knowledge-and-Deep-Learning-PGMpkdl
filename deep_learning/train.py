import atexit
import signal
import subprocess
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.metrics import *
from utils.losses import WeightedCrossEntropyLoss
from models.UNet import UNet
from utils.dataloader import CustomDataset
from config import config
import pandas as pd
# from ignite.metrics import Accuracy, Recall, Precision
from torchmetrics import Accuracy, Recall, Precision, F1Score
from torchmetrics import MetricCollection
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

now = datetime.now().strftime("%Y%m%d-%H%M")
logdir = rf'G:\train\logs\{now}'
writer = SummaryWriter(log_dir=logdir)
tb_path = r"C:\Users\xuejie\.conda\envs\zcenv\Scripts\tensorboard.exe"
tb_proc = subprocess.Popen([
                        tb_path,
                        f"--logdir={logdir}",
                        f"--port=6006"
                    ])
def cleanup():
    tb_proc.terminate()
    try:
        tb_proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        tb_proc.kill()
atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda sig, frame: exit(0))

def train_model(model, train_loader, val_loader, epochs, device):
    start = time.time()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    # criterion = WeightedCrossEntropyLoss(weight=config.loss_weight)
    criterion = nn.BCEWithLogitsLoss(weight=config.loss_weight)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config.factor, patience=config.patience, min_lr=1e-5)

    metrics = MetricCollection({
        'oa': Accuracy(task='binary'),
        'ua': Precision(task='binary'),
        'pa': Recall(task='binary'),
        'f1': F1Score(task='binary')
    }).to(device)
    train_metrics = metrics.clone()
    val_metrics = metrics.clone()

    history = []

    best_val_loss = float('inf')
    patience_counter = 0
    first = True
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_metrics.reset()

        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            if first:
                writer.add_graph(model, inputs)
                first = False

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_metrics.update(outputs, labels)


        train_loss = epoch_loss / len(train_loader)
        train_results = train_metrics.compute()

        # 验证阶段
        val_loss, val_results = evaluate(model, val_loader, device, criterion, val_metrics)

        record = {
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': train_loss,
            'train_oa': train_results['oa'].item(),
            'train_ua': train_results['ua'].item(),
            'train_pa': train_results['pa'].item(),
            'train_f1': train_results['f1'].item(),
            'val_loss': val_loss,
            'val_oa': val_results['oa'].item(),
            'val_ua': val_results['ua'].item(),
            'val_pa': val_results['pa'].item(),
            'val_f1': val_results['f1'].item(),
        }
        history.append(record)

        # 调整学习率
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Loss: {train_loss:.4f}(train)/{val_loss:.4f}(val) | "
            f"F1: {train_results['f1']:.4f}/{val_results['f1']:.4f} | "
            f"OA: {train_results['oa']:.4f}/{val_results['oa']:.4f} | "
            f"UA: {train_results['ua']:.4f}/{val_results['ua']:.4f} | "
            f"PA: {train_results['pa']:.4f}/{val_results['pa']:.4f} | "
            f"Time: {(time.time() - start) / 60:.1f} min"
        )

        # Tensorboard 监控
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)  # Loss
        # writer.add_scalar('Loss/train', train_loss, epoch)
        # writer.add_pr_curve('PR_Curve', labels, predictions, global_step=0)

        # EarlyStopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop:
                print("Early stopping triggered")
                break
        print(f"Early stopping count: {patience_counter}")

    for name, param in model.named_parameters():  # 权重分布
        writer.add_histogram(name, param, epoch)

    return history


def evaluate(model, data_loader, device, criterion, metrics):
    model.eval()
    total_loss = 0
    metrics.reset()

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            metrics.update(outputs, labels)

    return total_loss / len(data_loader), metrics.compute()