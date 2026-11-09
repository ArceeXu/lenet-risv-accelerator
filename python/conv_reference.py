
"""
conv_reference.py

项目：
    RISC-V SoC LeNet 手写数字识别硬件加速器

功能：
    1. NumPy手动实现Conv2d
    2. NumPy实现ReLU
    3. NumPy手动实现MaxPooling2d
    4. NumPy实现Flatten
    5. NumPy手动实现Linear
    6. 从export_weights.py导出的NPY文件读取权重
    7. 实现完整LeNet FP32前向传播
    8. 与PyTorch逐层比较
    9. 导出中间输出与误差报告

说明：
    本文件是FP32参考计算模型。
    不使用PyTorch算子完成NumPy推理。
    PyTorch仅用于加载模型和提供对照结果。

运行：
    python python/conv_reference.py
"""

from pathlib import Path
import csv
import json
import time

import numpy as np
import torch
from torchvision import datasets, transforms

from model import LeNet

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "data" / "checkpoints" / "lenet_fp32.pth"
)

WEIGHT_DIR = (
    ROOT / "data" / "weights" / "fp32" / "npy"
)

DATA_DIR = ROOT / "data" / "raw"

RESULT_DIR = (
    ROOT / "results" / "conv_reference"
)

OUTPUT_DIR = (
    RESULT_DIR / "intermediates"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_REPORT_PATH = (
    RESULT_DIR / "layer_error_report.csv"
)

PREDICTION_REPORT_PATH = (
    RESULT_DIR / "prediction_comparison.csv"
)

SUMMARY_PATH = (
    RESULT_DIR / "summary.json"
)

# 默认验证前10张测试图片
NUM_TEST_SAMPLES = 10

# 是否保存每张图片的逐层输出
SAVE_INTERMEDIATES = True

# FP32参考验证的数值容差
ATOL = 1e-4
RTOL = 1e-4

# 为保证不同设备结果相近，使用CPU对照
DEVICE = torch.device("cpu")

# LeNet各层名称，必须与model.py一致
LAYER_NAMES = [
    "conv1",
    "relu1",
    "pool1",
    "conv2",
    "relu2",
    "pool2",
    "flatten",
    "fc1",
    "relu3",
    "fc2",
    "relu4",
    "fc3"
]


def conv2d_numpy(
    x,
    weight,
    bias,
    stride=1,
    padding=0
):
    """
    手动实现二维卷积。

    输入：
        x:
            [Cin, Hin, Win]

        weight:
            [Cout, Cin, Kh, Kw]

        bias:
            [Cout]

        stride:
            卷积步长

        padding:
            零填充大小

    输出：
        y:
            [Cout, Hout, Wout]

    说明：
        实现的是深度学习中的互相关计算
        (cross-correlation)，与PyTorch Conv2d一致。

        不需要翻转卷积核。
    """
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)

    if x.ndim != 3:
        raise ValueError(
            f"Input must be [Cin,H,W], got {x.shape}"
        )

    if weight.ndim != 4:
        raise ValueError(
            "Weight must be [Cout,Cin,Kh,Kw]"
        )

    cin, hin, win = x.shape

    cout, weight_cin, kh, kw = weight.shape

    if cin != weight_cin:
        raise ValueError(
            f"Channel mismatch: {cin} vs {weight_cin}"
        )

    if bias.shape != (cout,):
        raise ValueError(
            f"Bias shape mismatch: {bias.shape}"
        )

    if stride <= 0 or padding < 0:
        raise ValueError(
            "Invalid stride or padding"
        )

    # 在输入周围补零
    if padding > 0:

        x_padded = np.pad(
            x,
            (
                (0, 0),
                (padding, padding),
                (padding, padding)
            ),
            mode="constant",
            constant_values=0
        )

    else:
        x_padded = x

    _, hp, wp = x_padded.shape

    if hp < kh or wp < kw:
        raise ValueError(
            "Kernel is larger than padded input"
        )

    # 输出尺寸
    hout = (hp - kh) // stride + 1
    wout = (wp - kw) // stride + 1

    # 输出特征图
    y = np.zeros(
        (cout, hout, wout),
        dtype=np.float32
    )

    # 六层循环实现卷积
    # 输出通道
    for oc in range(cout):

        # 输出行
        for oh in range(hout):

            # 输出列
            for ow in range(wout):

                # 从偏置初始化累加器
                acc = np.float32(bias[oc])

                # 输入通道
                for ic in range(cin):

                    # 卷积核行
                    for kr in range(kh):

                        # 卷积核列
                        for kc in range(kw):

                            ih = oh * stride + kr
                            iw = ow * stride + kc

                            input_value = x_padded[
                                ic, ih, iw
                            ]

                            weight_value = weight[
                                oc, ic, kr, kc
                            ]

                            # 显式使用float32乘加
                            product = np.float32(
                                input_value * weight_value
                            )

                            acc = np.float32(
                                acc + product
                            )

                y[oc, oh, ow] = acc

    return y


def relu_numpy(x):
    """
    ReLU(x) = max(0, x)
    """

    x = np.asarray(x, dtype=np.float32)

    return np.maximum(
        x,
        np.float32(0)
    ).astype(np.float32)


def maxpool2d_numpy(
    x,
    kernel_size=2,
    stride=2
):
    """
    输入：
        [C,H,W]

    输出：
        [C,Hout,Wout]

    默认：
        2x2最大池化
        stride=2
    """

    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 3:
        raise ValueError(
            "MaxPool input must be [C,H,W]"
        )

    if kernel_size <= 0 or stride <= 0:
        raise ValueError(
            "Invalid pooling parameters"
        )

    channels, height, width = x.shape

    if height < kernel_size or width < kernel_size:
        raise ValueError(
            "Pooling window larger than input"
        )

    hout = (
        (height - kernel_size) // stride + 1
    )

    wout = (
        (width - kernel_size) // stride + 1
    )

    y = np.zeros(
        (channels, hout, wout),
        dtype=np.float32
    )

    for c in range(channels):

        for oh in range(hout):

            for ow in range(wout):

                h_start = oh * stride
                w_start = ow * stride

                # 初始化为窗口第一个元素
                max_value = x[
                    c,
                    h_start,
                    w_start
                ]

                for kh in range(kernel_size):

                    for kw in range(kernel_size):

                        value = x[
                            c,
                            h_start + kh,
                            w_start + kw
                        ]

                        if value > max_value:
                            max_value = value

                y[c, oh, ow] = max_value

    return y


def flatten_numpy(x):
    """
    PyTorch:
        torch.flatten(x, start_dim=1)

    单样本输入：
        [C,H,W]

    输出：
        [C*H*W]

    必须使用C-order。
    """

    x = np.asarray(x, dtype=np.float32)

    return x.reshape(-1, order="C").copy()


def linear_numpy(x, weight, bias):
    """
    输入：
        x:      [Nin]
        weight: [Nout,Nin]
        bias:   [Nout]

    输出：
        y: [Nout]

    计算：
        y[o] = bias[o] + sum(weight[o,i] * x[i])
    """

    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)

    if x.ndim != 1:
        raise ValueError(
            "Linear input must be 1D"
        )

    if weight.ndim != 2:
        raise ValueError(
            "Linear weight must be 2D"
        )

    nout, nin = weight.shape

    if x.shape[0] != nin:
        raise ValueError(
            f"Linear input mismatch: "
            f"{x.shape[0]} vs {nin}"
        )

    if bias.shape != (nout,):
        raise ValueError(
            "Linear bias shape mismatch"
        )

    y = np.zeros(
        nout,
        dtype=np.float32
    )

    for o in range(nout):

        acc = np.float32(bias[o])

        for i in range(nin):

            product = np.float32(
                x[i] * weight[o, i]
            )

            acc = np.float32(
                acc + product
            )

        y[o] = acc

    return y


# 加载导出的FP32权重
def load_exported_weights():

    parameter_names = [
        "conv1_weight",
        "conv1_bias",
        "conv2_weight",
        "conv2_bias",
        "fc1_weight",
        "fc1_bias",
        "fc2_weight",
        "fc2_bias",
        "fc3_weight",
        "fc3_bias"
    ]

    weights = {}

    for name in parameter_names:

        path = WEIGHT_DIR / f"{name}.npy"

        if not path.exists():
            raise FileNotFoundError(
                f"Weight not found: {path}\n"
                "Please run export_weights.py first."
            )

        weights[name] = np.load(
            path,
            allow_pickle=False
        ).astype(np.float32)

    # 检查权重形状
    expected_shapes = {
        "conv1_weight": (6, 1, 5, 5),
        "conv1_bias": (6,),
        "conv2_weight": (16, 6, 5, 5),
        "conv2_bias": (16,),
        "fc1_weight": (120, 256),
        "fc1_bias": (120,),
        "fc2_weight": (84, 120),
        "fc2_bias": (84,),
        "fc3_weight": (10, 84),
        "fc3_bias": (10,)
    }

    for name, expected_shape in expected_shapes.items():

        actual_shape = weights[name].shape

        if actual_shape != expected_shape:
            raise ValueError(
                f"{name}: expected {expected_shape}, "
                f"got {actual_shape}"
            )

    print(f"Loaded {len(weights)} parameter arrays.")

    return weights


# NumPy完整LeNet前向传播
def lenet_forward_numpy(x, weights):
    """
    输入：
        x: [1,28,28]

    返回：
        logits: [10]
        outputs: 各层中间结果字典
    """

    x = np.asarray(x, dtype=np.float32)

    if x.shape != (1, 28, 28):
        raise ValueError(
            f"Expected [1,28,28], got {x.shape}"
        )

    outputs = {}

    # --------------------------------------------------------
    # Conv1: [1,28,28] -> [6,24,24]
    # --------------------------------------------------------

    x = conv2d_numpy(
        x,
        weights["conv1_weight"],
        weights["conv1_bias"],
        stride=1,
        padding=0
    )

    outputs["conv1"] = x.copy()

    # ReLU1
    x = relu_numpy(x)
    outputs["relu1"] = x.copy()

    # Pool1: [6,24,24] -> [6,12,12]
    x = maxpool2d_numpy(x, 2, 2)
    outputs["pool1"] = x.copy()

    # --------------------------------------------------------
    # Conv2: [6,12,12] -> [16,8,8]
    # --------------------------------------------------------

    x = conv2d_numpy(
        x,
        weights["conv2_weight"],
        weights["conv2_bias"],
        stride=1,
        padding=0
    )

    outputs["conv2"] = x.copy()

    # ReLU2
    x = relu_numpy(x)
    outputs["relu2"] = x.copy()

    # Pool2: [16,8,8] -> [16,4,4]
    x = maxpool2d_numpy(x, 2, 2)
    outputs["pool2"] = x.copy()

    # --------------------------------------------------------
    # Flatten: [16,4,4] -> [256]
    # --------------------------------------------------------

    x = flatten_numpy(x)
    outputs["flatten"] = x.copy()

    # --------------------------------------------------------
    # FC1: [256] -> [120]
    # --------------------------------------------------------

    x = linear_numpy(
        x,
        weights["fc1_weight"],
        weights["fc1_bias"]
    )

    outputs["fc1"] = x.copy()

    x = relu_numpy(x)
    outputs["relu3"] = x.copy()

    # --------------------------------------------------------
    # FC2: [120] -> [84]
    # --------------------------------------------------------

    x = linear_numpy(
        x,
        weights["fc2_weight"],
        weights["fc2_bias"]
    )

    outputs["fc2"] = x.copy()

    x = relu_numpy(x)
    outputs["relu4"] = x.copy()

    # --------------------------------------------------------
    # FC3: [84] -> [10]
    # --------------------------------------------------------

    x = linear_numpy(
        x,
        weights["fc3_weight"],
        weights["fc3_bias"]
    )

    outputs["fc3"] = x.copy()

    return x, outputs


# 加载PyTorch对照模型
def load_pytorch_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Please run train.py first."
        )

    model = LeNet().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# 加载MNIST测试数据
def load_test_dataset():

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform
    )

    return dataset


# 单层误差比较
def compare_layer(
    sample_index,
    layer_name,
    numpy_output,
    torch_output
):
    """
    返回每层误差统计信息。

    因为FP32加法顺序和硬件实现不同，
    结果不要求逐位相等。

    使用atol + rtol * |reference|
    判断是否在容差范围内。
    """

    numpy_output = np.asarray(
        numpy_output,
        dtype=np.float32
    )

    torch_output = (
        torch_output.detach()
        .cpu()
        .numpy()
        .squeeze(0)
        .astype(np.float32)
    )

    if numpy_output.shape != torch_output.shape:

        raise ValueError(
            f"Shape mismatch at {layer_name}: "
            f"NumPy={numpy_output.shape}, "
            f"PyTorch={torch_output.shape}"
        )

    abs_error = np.abs(
        numpy_output.astype(np.float64)
        - torch_output.astype(np.float64)
    )

    max_abs_error = float(
        np.max(abs_error)
    )

    mean_abs_error = float(
        np.mean(abs_error)
    )

    passed = np.allclose(
        numpy_output,
        torch_output,
        atol=ATOL,
        rtol=RTOL
    )

    return {
        "sample_index": sample_index,
        "layer": layer_name,
        "shape": str(numpy_output.shape),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "passed": bool(passed)
    }


# 保存逐层输出
def save_intermediate_outputs(
    sample_index,
    numpy_outputs
):

    sample_dir = (
        OUTPUT_DIR / f"sample_{sample_index:04d}"
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for layer_name, array in numpy_outputs.items():

        np.save(
            sample_dir / f"{layer_name}.npy",
            array
        )


# 保存CSV和JSON报告
def save_reports(
    layer_results,
    prediction_results,
    summary
):

    # 保存逐层误差
    with open(
        ERROR_REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "layer",
                "shape",
                "max_abs_error",
                "mean_abs_error",
                "passed"
            ]
        )

        writer.writeheader()
        writer.writerows(layer_results)

    # 保存预测结果
    with open(
        PREDICTION_REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "true_label",
                "numpy_prediction",
                "pytorch_prediction",
                "prediction_match"
            ]
        )

        writer.writeheader()
        writer.writerows(prediction_results)

    # 保存总结
    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4
        )


def main():

    print("=" * 65)
    print("LeNet NumPy FP32 Reference Verification")
    print("=" * 65)

    print(f"Model Path : {MODEL_PATH}")
    print(f"Weight Dir : {WEIGHT_DIR}")
    print(f"Device     : {DEVICE}")

    # 加载导出权重
    weights = load_exported_weights()

    # 加载PyTorch模型
    model = load_pytorch_model()

    # 确认导出权重与checkpoint一致
    state_dict = model.state_dict()

    for name, array in weights.items():

        torch_name = name.replace("_", ".", 1)

        checkpoint_array = (
            state_dict[torch_name]
            .detach()
            .cpu()
            .numpy()
        )

        if not np.array_equal(
            array,
            checkpoint_array
        ):
            raise AssertionError(
                f"Exported weight differs: {name}"
            )

    print("Exported weights match checkpoint.")

    dataset = load_test_dataset()

    sample_count = min(
        NUM_TEST_SAMPLES,
        len(dataset)
    )

    layer_results = []
    prediction_results = []

    num_prediction_matches = 0
    num_layer_passes = 0
    total_layer_checks = 0

    start_time = time.perf_counter()

    # 逐张图片验证
    for sample_index in range(sample_count):

        image, label = dataset[sample_index]

        # PyTorch输入形状 [1,1,28,28]
        torch_input = (
            image.unsqueeze(0).to(DEVICE)
        )

        # NumPy输入形状 [1,28,28]
        numpy_input = (
            image.numpy().astype(np.float32)
        )

        # NumPy前向传播
        numpy_logits, numpy_outputs = (
            lenet_forward_numpy(
                numpy_input,
                weights
            )
        )

        # PyTorch前向传播
        with torch.no_grad():

            torch_outputs = (
                model.forward_with_intermediates(
                    torch_input
                )
            )

        # 比较每个中间层
        for layer_name in LAYER_NAMES:

            result = compare_layer(
                sample_index,
                layer_name,
                numpy_outputs[layer_name],
                torch_outputs[layer_name]
            )

            layer_results.append(result)

            total_layer_checks += 1

            if result["passed"]:
                num_layer_passes += 1

        # 预测类别
        numpy_prediction = int(
            np.argmax(numpy_logits)
        )

        pytorch_prediction = int(
            torch.argmax(
                torch_outputs["fc3"],
                dim=1
            ).item()
        )

        match = (
            numpy_prediction == pytorch_prediction
        )

        if match:
            num_prediction_matches += 1

        prediction_results.append({
            "sample_index": sample_index,
            "true_label": int(label),
            "numpy_prediction": numpy_prediction,
            "pytorch_prediction": pytorch_prediction,
            "prediction_match": bool(match)
        })

        if SAVE_INTERMEDIATES:

            save_intermediate_outputs(
                sample_index,
                numpy_outputs
            )

        # 统计当前样本通过的层数
        sample_layer_results = [
            item for item in layer_results
            if item["sample_index"] == sample_index
        ]

        sample_passed = all(
            item["passed"]
            for item in sample_layer_results
        )

        status = (
            "PASS" if sample_passed and match
            else "FAIL"
        )

        print(
            f"Sample {sample_index:04d} | "
            f"Label={label} | "
            f"NumPy={numpy_prediction} | "
            f"PyTorch={pytorch_prediction} | "
            f"{status}"
        )

    elapsed_time = (
        time.perf_counter() - start_time
    )

    all_layers_passed = (
        num_layer_passes == total_layer_checks
    )

    all_predictions_match = (
        num_prediction_matches == sample_count
    )

    summary = {
        "model": "LeNet",
        "precision": "FP32",
        "reference": "PyTorch",
        "implementation": "NumPy manual loops",
        "samples_tested": sample_count,
        "layers_per_sample": len(LAYER_NAMES),
        "total_layer_checks": total_layer_checks,
        "passed_layer_checks": num_layer_passes,
        "all_layers_passed": all_layers_passed,
        "prediction_matches": num_prediction_matches,
        "all_predictions_match": all_predictions_match,
        "atol": ATOL,
        "rtol": RTOL,
        "elapsed_seconds": elapsed_time
    }

    save_reports(
        layer_results,
        prediction_results,
        summary
    )

    print("\n" + "=" * 65)
    print("Verification Summary")
    print("=" * 65)

    print(f"Samples Tested     : {sample_count}")

    print(
        f"Layer Checks       : "
        f"{num_layer_passes}/{total_layer_checks}"
    )

    print(
        f"Prediction Matches : "
        f"{num_prediction_matches}/{sample_count}"
    )

    print(
        f"Execution Time     : "
        f"{elapsed_time:.2f} seconds"
    )

    print(f"Error Report       : {ERROR_REPORT_PATH}")
    print(f"Prediction Report  : {PREDICTION_REPORT_PATH}")

    if all_layers_passed and all_predictions_match:

        print("\nALL REFERENCE TESTS PASSED!")

    else:

        print("\nWARNING: Some reference tests failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
