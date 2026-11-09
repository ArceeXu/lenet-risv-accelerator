
"""
evaluate.py

项目：
    RISC-V SoC LeNet 手写数字识别硬件加速器

功能：
    1. 加载训练完成的 LeNet FP32 模型
    2. 评估 MNIST 测试集准确率
    3. 计算交叉熵损失
    4. 计算混淆矩阵
    5. 计算每个数字类别的准确率
    6. 绘制混淆矩阵
    7. 绘制预测结果示例
    8. 导出测试图片及标签
    9. 导出各层中间输出
   10. 为后续 Verilog 仿真提供 FP32 参考数据

运行：
    python python/evaluate.py

依赖：
    torch
    torchvision
    numpy
    matplotlib
    tqdm
"""

from pathlib import Path
import csv
import json
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import LeNet

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw"

MODEL_PATH = (
    ROOT / "data" / "checkpoints" / "lenet_fp32.pth"
)

RESULT_DIR = ROOT / "results" / "accuracy"

VECTOR_DIR = ROOT / "data" / "test_vectors"

INTERMEDIATE_DIR = (
    VECTOR_DIR / "fp32_intermediates"
)

# 自动创建输出目录
RESULT_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件
METRICS_PATH = RESULT_DIR / "evaluation_metrics.json"

CLASS_ACCURACY_PATH = (
    RESULT_DIR / "class_accuracy.csv"
)

CONFUSION_MATRIX_PATH = (
    RESULT_DIR / "confusion_matrix.csv"
)

CONFUSION_FIGURE_PATH = (
    RESULT_DIR / "confusion_matrix.png"
)

PREDICTION_FIGURE_PATH = (
    RESULT_DIR / "prediction_examples.png"
)

PREDICTIONS_CSV_PATH = (
    RESULT_DIR / "test_predictions.csv"
)

# 参数配置
BATCH_SIZE = 128

NUM_WORKERS = 0

SEED = 42

# 导出多少张测试图片供后续硬件验证
NUM_EXPORT_SAMPLES = 100

# 绘制多少张预测结果示例
NUM_SHOW_SAMPLES = 20

# 是否导出中间层结果
EXPORT_INTERMEDIATES = True

# 自动选择CPU或GPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# 固定随机种子
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 加载MNIST测试集
def load_test_dataset():
    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    print("\nDataset Information")
    print("-" * 60)
    print(f"Test Samples : {len(test_dataset)}")
    print(f"Batch Size   : {BATCH_SIZE}")

    return test_dataset, test_loader


# 加载训练好的模型
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}\n"
            "Please run train.py first."
        )

    model = LeNet().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    print("\nModel Information")
    print("-" * 60)
    print(f"Model Path : {MODEL_PATH}")
    print(f"Device     : {DEVICE}")

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Parameters : {total_params}")

    return model


# 测试集完整评估
@torch.no_grad()
def evaluate_model(model, test_loader):

    model.eval()

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    )

    total_loss = 0.0

    total_correct = 0
    total_samples = 0

    # 混淆矩阵：
    # 行表示真实类别，列表示预测类别
    confusion_matrix = np.zeros(
        (10, 10),
        dtype=np.int64
    )

    # 保存每张图片的预测结果
    all_labels = []
    all_predictions = []
    all_confidences = []

    progress_bar = tqdm(
        test_loader,
        desc="Evaluating MNIST"
    )

    for images, labels in progress_bar:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # 前向传播
        outputs = model(images)

        # 交叉熵损失
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        # 预测类别
        predictions = outputs.argmax(dim=1)

        # Softmax仅用于计算置信度
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences = probabilities.max(
            dim=1
        ).values

        # 正确数量
        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

        # 转换为CPU NumPy
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        confidences_np = confidences.cpu().numpy()

        all_labels.extend(
            labels_np.tolist()
        )

        all_predictions.extend(
            predictions_np.tolist()
        )

        all_confidences.extend(
            confidences_np.tolist()
        )

        # 更新混淆矩阵
        for true_label, pred_label in zip(
            labels_np,
            predictions_np
        ):
            confusion_matrix[
                int(true_label),
                int(pred_label)
            ] += 1

        progress_bar.set_postfix(
            accuracy=(
                f"{100 * total_correct / total_samples:.2f}%"
            )
        )

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_samples,
        "confusion_matrix": confusion_matrix,
        "labels": all_labels,
        "predictions": all_predictions,
        "confidences": all_confidences
    }


# 计算每个类别的准确率
def calculate_class_accuracy(confusion_matrix):

    class_results = []

    for digit in range(10):

        total = int(
            confusion_matrix[digit, :].sum()
        )

        correct = int(
            confusion_matrix[digit, digit]
        )

        accuracy = (
            correct / total
            if total > 0 else 0.0
        )

        class_results.append({
            "digit": digit,
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        })

    return class_results


# 保存完整评估结果
def save_evaluation_results(results, class_results):

    metrics = {
        "model": "LeNet",
        "precision": "FP32",
        "dataset": "MNIST Test",
        "total_samples": results["total"],
        "correct_samples": results["correct"],
        "test_loss": results["loss"],
        "test_accuracy": results["accuracy"],
        "test_accuracy_percent": (
            results["accuracy"] * 100
        )
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
            ensure_ascii=False
        )

    # 保存每类准确率
    with open(
        CLASS_ACCURACY_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "digit",
            "correct",
            "total",
            "accuracy",
            "accuracy_percent"
        ])

        for item in class_results:

            writer.writerow([
                item["digit"],
                item["correct"],
                item["total"],
                item["accuracy"],
                item["accuracy"] * 100
            ])

    # 保存混淆矩阵
    np.savetxt(
        CONFUSION_MATRIX_PATH,
        results["confusion_matrix"],
        delimiter=",",
        fmt="%d"
    )

    # 保存每张图片的预测
    with open(
        PREDICTIONS_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "sample_index",
            "true_label",
            "predicted_label",
            "confidence",
            "correct"
        ])

        for index, (label, pred, confidence) in enumerate(
            zip(
                results["labels"],
                results["predictions"],
                results["confidences"]
            )
        ):

            writer.writerow([
                index,
                label,
                pred,
                confidence,
                int(label == pred)
            ])

    print("\nEvaluation results saved.")


# 绘制混淆矩阵
def plot_confusion_matrix(confusion_matrix):

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        confusion_matrix,
        interpolation="nearest",
        cmap="Blues"
    )

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xlabel="Predicted Label",
        ylabel="True Label",
        title="LeNet MNIST Confusion Matrix"
    )

    threshold = (
        confusion_matrix.max() / 2
    )

    for i in range(10):
        for j in range(10):

            value = confusion_matrix[i, j]

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
                fontsize=8
            )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# 绘制预测结果示例
@torch.no_grad()
def plot_prediction_examples(
    model,
    test_dataset
):

    model.eval()

    count = min(
        NUM_SHOW_SAMPLES,
        len(test_dataset)
    )

    # 固定随机采样，方便复现实验
    rng = np.random.default_rng(SEED)

    indices = rng.choice(
        len(test_dataset),
        size=count,
        replace=False
    )

    columns = 5
    rows = (count + columns - 1) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 2.5 * rows),
        squeeze=False
    )

    for position, index in enumerate(indices):

        image, label = test_dataset[
            int(index)
        ]

        input_tensor = (
            image.unsqueeze(0).to(DEVICE)
        )

        output = model(input_tensor)

        prediction = int(
            output.argmax(dim=1).item()
        )

        ax = axes[
            position // columns,
            position % columns
        ]

        ax.imshow(
            image.squeeze(0).numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.set_title(
            f"True: {label} | Pred: {prediction}",
            color=(
                "green"
                if prediction == label
                else "red"
            ),
            fontsize=10
        )

        ax.axis("off")

    # 隐藏多余的子图
    for position in range(count, rows * columns):

        axes[
            position // columns,
            position % columns
        ].axis("off")

    plt.tight_layout()

    plt.savefig(
        PREDICTION_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# 导出FPGA测试图片与中间输出
@torch.no_grad()
def export_test_vectors(
    model,
    test_dataset
):

    model.eval()

    export_count = min(
        NUM_EXPORT_SAMPLES,
        len(test_dataset)
    )

    print("\nExporting test vectors...")
    print(f"Samples: {export_count}")

    # 为了保证可重复性：
    # 默认选择测试集前100张图片
    #
    # 与 test_predictions.csv 中的
    # sample_index 一一对应

    manifest = []

    for index in tqdm(
        range(export_count),
        desc="Exporting"
    ):

        image, label = test_dataset[index]

        input_tensor = (
            image.unsqueeze(0).to(DEVICE)
        )

        # 标准前向传播
        logits = model(input_tensor)

        prediction = int(
            logits.argmax(dim=1).item()
        )

        # 保存原始图像（0-255）
        image_uint8 = (
            image.squeeze(0).numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)

        # 保存归一化输入（FP32）
        image_fp32 = (
            image.squeeze(0).numpy()
            .astype(np.float32)
        )

        sample_name = f"sample_{index:04d}"

        # 每个样本一个文件夹
        sample_dir = (
            VECTOR_DIR / sample_name
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            sample_dir / "input_uint8.npy",
            image_uint8
        )

        np.save(
            sample_dir / "input_fp32.npy",
            image_fp32
        )

        # 保存FP32 logits
        np.save(
            sample_dir / "logits_fp32.npy",
            logits.cpu().numpy().squeeze(0)
        )

        # 保存标签信息
        label_info = {
            "sample_index": index,
            "true_label": int(label),
            "predicted_label": prediction,
            "correct": bool(prediction == label)
        }

        with open(
            sample_dir / "label.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                label_info,
                f,
                indent=4
            )

        # 保存各层FP32中间输出
        if EXPORT_INTERMEDIATES:

            outputs = (
                model.forward_with_intermediates(
                    input_tensor
                )
            )

            intermediate_sample_dir = (
                INTERMEDIATE_DIR / sample_name
            )

            intermediate_sample_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for layer_name, tensor in outputs.items():

                array = (
                    tensor.detach()
                    .cpu()
                    .numpy()
                    .squeeze(0)
                    .astype(np.float32)
                )

                np.save(
                    intermediate_sample_dir
                    / f"{layer_name}.npy",
                    array
                )

        manifest.append({
            "sample_index": index,
            "true_label": int(label),
            "predicted_label": prediction,
            "correct": int(prediction == label)
        })

    # 保存测试样本清单
    manifest_path = (
        VECTOR_DIR / "test_vectors_manifest.csv"
    )

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "true_label",
                "predicted_label",
                "correct"
            ]
        )

        writer.writeheader()
        writer.writerows(manifest)

    print(f"Test vectors saved to: {VECTOR_DIR}")


# 打印评估报告
def print_evaluation_report(
    results,
    class_results
):

    print("\n" + "=" * 60)
    print("LeNet MNIST Evaluation Report")
    print("=" * 60)

    print(f"Device          : {DEVICE}")
    print(f"Test Samples    : {results['total']}")
    print(f"Correct Samples : {results['correct']}")
    print(f"Test Loss       : {results['loss']:.6f}")

    print(
        f"Test Accuracy   : "
        f"{results['accuracy'] * 100:.2f}%"
    )

    print("\nPer-Class Accuracy")
    print("-" * 60)

    print(
        f"{'Digit':<10}"
        f"{'Correct':<12}"
        f"{'Total':<12}"
        f"{'Accuracy':<12}"
    )

    for item in class_results:

        print(
            f"{item['digit']:<10}"
            f"{item['correct']:<12}"
            f"{item['total']:<12}"
            f"{item['accuracy'] * 100:.2f}%"
        )

    print("\n" + "=" * 60)


def main():

    set_seed(SEED)

    print("=" * 60)
    print("LeNet FP32 Evaluation")
    print("=" * 60)

    # 1. 加载测试集
    test_dataset, test_loader = (
        load_test_dataset()
    )

    # 2. 加载最佳模型
    model = load_model()

    # 3. 完整测试集评估
    results = evaluate_model(
        model,
        test_loader
    )

    # 4. 各类别准确率
    class_results = calculate_class_accuracy(
        results["confusion_matrix"]
    )

    # 5. 保存评估结果
    save_evaluation_results(
        results,
        class_results
    )

    # 6. 混淆矩阵
    plot_confusion_matrix(
        results["confusion_matrix"]
    )

    # 7. 预测结果示例
    plot_prediction_examples(
        model,
        test_dataset
    )

    # 8. 导出硬件测试向量
    export_test_vectors(
        model,
        test_dataset
    )

    # 9. 打印报告
    print_evaluation_report(
        results,
        class_results
    )

    print("\nOutput Files:")
    print(f"Metrics        : {METRICS_PATH}")
    print(f"Class Accuracy : {CLASS_ACCURACY_PATH}")
    print(f"Confusion CSV  : {CONFUSION_MATRIX_PATH}")
    print(f"Confusion Plot : {CONFUSION_FIGURE_PATH}")
    print(f"Predictions    : {PREDICTION_FIGURE_PATH}")
    print(f"Test Vectors   : {VECTOR_DIR}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
