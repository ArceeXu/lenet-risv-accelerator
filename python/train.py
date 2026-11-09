
"""
train.py

项目：RISC-V SoC LeNet 手写数字识别硬件加速器

功能：
1. 加载 MNIST 数据集
2. 训练 LeNet 模型
3. 验证模型准确率
4. 保存最佳模型权重
5. 记录训练日志
6. 绘制 Loss / Accuracy 曲线
7. 使用测试集评估最终模型

运行：
    python python/train.py

依赖：
    torch
    torchvision
    numpy
    matplotlib
    tqdm
"""

from pathlib import Path
import csv
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from model import LeNet

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw"
CHECKPOINT_DIR = ROOT / "data" / "checkpoints"
RESULT_DIR = ROOT / "results" / "accuracy"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = CHECKPOINT_DIR / "lenet_fp32.pth"
LOG_PATH = RESULT_DIR / "training_log.csv"
FIGURE_PATH = RESULT_DIR / "training_curves.png"

# 训练参数
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# 数据集划分
TRAIN_RATIO = 0.9

# 随机种子
SEED = 42

# Windows 环境建议先设置为0
NUM_WORKERS = 0

# 自动选择计算设备
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 设置随机种子
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 尽可能提高可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 加载数据集
def load_datasets():

    # 保持输入在[0,1]范围
    # 便于后续INT8量化及FPGA输入数据处理
    transform = transforms.ToTensor()

    full_train_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform
    )

    train_size = int(
        len(full_train_dataset) * TRAIN_RATIO
    )

    val_size = (
        len(full_train_dataset) - train_size
    )

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    print("\nDataset Information")
    print("-" * 50)
    print(f"Training samples   : {train_size}")
    print(f"Validation samples : {val_size}")
    print(f"Test samples       : {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} Training",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # 清空梯度
        optimizer.zero_grad(set_to_none=True)

        # 前向传播
        outputs = model(images)

        # 计算交叉熵损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        # 统计损失
        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        # 统计准确率
        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        progress_bar.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{100.0 * correct / total:.2f}%"
        )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# 验证 / 测试模型
@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# 6. 保存训练日志
def save_training_log(history):

    with open(
        LOG_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
            "epoch_time_seconds"
        ])

        for i in range(len(history["epoch"])):

            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
                history["lr"][i],
                history["epoch_time"][i]
            ])

    print(f"Training log saved: {LOG_PATH}")


# 绘制训练曲线
def plot_training_curves(history):

    epochs = history["epoch"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    # Loss
    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
        marker="o"
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
        marker="s"
    )

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Accuracy
    axes[1].plot(
        epochs,
        np.array(history["train_acc"]) * 100,
        label="Train Accuracy",
        marker="o"
    )

    axes[1].plot(
        epochs,
        np.array(history["val_acc"]) * 100,
        label="Validation Accuracy",
        marker="s"
    )

    axes[1].set_title(
        "Training and Validation Accuracy"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Training curves saved: {FIGURE_PATH}")


def main():

    set_seed(SEED)

    print("=" * 60)
    print("LeNet MNIST Training")
    print("=" * 60)

    print(f"Device       : {DEVICE}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Batch Size   : {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Random Seed  : {SEED}")

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # 加载数据
    train_loader, val_loader, test_loader = (
        load_datasets()
    )

    # 创建模型
    model = LeNet().to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"\nModel Parameters: {total_params}")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    # 训练历史
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_time": []
    }

    best_val_accuracy = 0.0
    best_epoch = 0

    total_start_time = time.perf_counter()
    
    # Epoch循环
    for epoch in range(1, EPOCHS + 1):

        epoch_start_time = time.perf_counter()

        current_lr = optimizer.param_groups[0]["lr"]

        # 训练
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
            epoch
        )

        # 验证
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        epoch_time = (
            time.perf_counter() - epoch_start_time
        )

        # 保存历史数据
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        # 保存验证集表现最好的模型
        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                f"  Best model saved "
                f"(Val Acc: {val_acc * 100:.2f}%)"
            )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # 更新学习率
        scheduler.step()

        # 每轮保存日志
        save_training_log(history)

    total_time = (
        time.perf_counter() - total_start_time
    )

    
    # 加载最佳模型
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    # 使用独立测试集进行最终评估
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print(f"Best Epoch       : {best_epoch}")
    print(
        f"Best Val Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(f"Test Loss        : {test_loss:.4f}")
    print(
        f"Test Accuracy    : "
        f"{test_acc * 100:.2f}%"
    )

    print(
        f"Total Train Time : "
        f"{total_time:.1f}s"
    )

    # 保存曲线
    plot_training_curves(history)

    print("\nSaved Files:")
    print(f"Model  : {MODEL_PATH}")
    print(f"Log    : {LOG_PATH}")
    print(f"Figure : {FIGURE_PATH}")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
