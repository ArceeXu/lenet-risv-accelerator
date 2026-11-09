
"""
export_weights.py

项目：
    RISC-V SoC LeNet 手写数字识别硬件加速器

功能：
    1. 加载训练好的 lenet_fp32.pth
    2. 导出各层 FP32 权重和偏置（.npy）
    3. 导出十六进制 IEEE754 FP32 数据（.mem）
    4. 导出十进制文本（.txt）
    5. 导出 C 语言数组头文件（.h）
    6. 保存参数形状、数量和排列顺序（.json）
    7. 重新加载导出数据，验证与原模型完全一致

注意：
    本程序不执行 INT8 量化。
    .mem 中存储的是浮点数的 IEEE754 位模式，
    不是 INT8 数据，也不是十六进制整数权重。

运行：
    python python/export_weights.py
"""

from pathlib import Path
import json
import re

import numpy as np
import torch

from model import LeNet

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "data" / "checkpoints" / "lenet_fp32.pth"
)

EXPORT_DIR = (
    ROOT / "data" / "weights" / "fp32"
)

NPY_DIR = EXPORT_DIR / "npy"
MEM_DIR = EXPORT_DIR / "mem"
TXT_DIR = EXPORT_DIR / "txt"
C_DIR = EXPORT_DIR / "c"

MANIFEST_PATH = EXPORT_DIR / "weights_manifest.json"

for directory in [
    NPY_DIR,
    MEM_DIR,
    TXT_DIR,
    C_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# 导出配置
# 权重和偏置都采用 FP32
DTYPE = np.float32

# 只导出这些层
LAYER_NAMES = [
    "conv1",
    "conv2",
    "fc1",
    "fc2",
    "fc3"
]

# 导出C语言头文件时，每行多少个数
C_VALUES_PER_LINE = 6

# 输出文件中的十进制浮点精度
FLOAT_FORMAT = "%.9g"


# 加载模型
def load_trained_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Cannot find model:\n{MODEL_PATH}\n"
            "Please run train.py first."
        )

    # 在CPU上导出，避免依赖CUDA环境
    device = torch.device("cpu")

    model = LeNet().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    print("=" * 65)
    print("LeNet FP32 Weight Export")
    print("=" * 65)

    print(f"Model : {MODEL_PATH}")
    print(f"Output: {EXPORT_DIR}")

    return model


# 保存 NumPy 文件
def save_npy(name, array):

    path = NPY_DIR / f"{name}.npy"

    np.save(
        path,
        array
    )

    return path


# 保存十进制 TXT 文件
def save_txt(name, array):

    path = TXT_DIR / f"{name}.txt"

    # 统一按 C-order 展平
    flattened = array.flatten(order="C")

    np.savetxt(
        path,
        flattened,
        fmt=FLOAT_FORMAT
    )

    return path


# 保存 IEEE754 十六进制 MEM 文件
def save_mem(name, array):

    path = MEM_DIR / f"{name}.mem"

    # 确保数据为小端主机上的原生float32
    fp32_array = np.asarray(
        array,
        dtype=np.float32
    )

    flattened = np.ascontiguousarray(
        fp32_array.flatten(order="C")
    )

    # 将float32的32位二进制模式解释为uint32
    bit_patterns = flattened.view(np.uint32)

    with open(
        path,
        "w",
        encoding="ascii"
    ) as f:

        for bits in bit_patterns:

            f.write(
                f"{int(bits):08X}\n"
            )

    return path


# 将数组导出为C语言头文件
def make_c_identifier(name):

    return re.sub(
        r"[^a-zA-Z0-9_]",
        "_",
        name
    )


def save_c_header(name, array):

    path = C_DIR / f"{name}.h"

    identifier = make_c_identifier(name)

    macro_name = identifier.upper()

    flattened = array.flatten(order="C")

    # 生成C语言浮点常量
    values = []

    for value in flattened:

        # 9位有效数字足以往返还原float32
        formatted = format(
            float(value),
            ".9g"
        )

        # 确保整数形式也写为浮点数
        if (
            "." not in formatted
            and "e" not in formatted.lower()
        ):
            formatted += ".0"

        values.append(
            formatted + "f"
        )

    lines = []

    for i in range(
        0,
        len(values),
        C_VALUES_PER_LINE
    ):

        lines.append(
            "    " +
            ", ".join(
                values[i:i + C_VALUES_PER_LINE]
            )
        )

    contents = ",\n".join(lines)

    shape = list(array.shape)

    shape_text = ", ".join(
        str(dim) for dim in shape
    )

    header_guard = (
        f"LENET_{macro_name}_H"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"#ifndef {header_guard}\n"
        )

        f.write(
            f"#define {header_guard}\n\n"
        )

        f.write(
            f"#define {macro_name}_SIZE "
            f"{flattened.size}\n"
        )

        f.write(
            f"#define {macro_name}_NDIM "
            f"{array.ndim}\n\n"
        )

        f.write(
            f"/* Shape: [{shape_text}] */\n"
        )

        f.write(
            "/* Flatten order: C (row-major) */\n"
        )

        f.write(
            f"static const float "
            f"{identifier}[{flattened.size}] = {{\n"
        )

        f.write(contents)

        f.write(
            "\n};\n\n"
        )

        f.write(
            f"#endif /* {header_guard} */\n"
        )

    return path


# 计算各层参数统计信息
def get_array_statistics(array):

    flattened = array.flatten()

    return {
        "min": float(flattened.min()),
        "max": float(flattened.max()),
        "mean": float(flattened.mean()),
        "std": float(flattened.std()),
        "num_elements": int(flattened.size),
        "num_bytes": int(flattened.nbytes)
    }


# 导出单个参数
def export_parameter(name, tensor):

    array = (
        tensor.detach()
        .cpu()
        .numpy()
        .astype(DTYPE)
        .copy()
    )

    # 保存不同格式
    npy_path = save_npy(name, array)

    txt_path = save_txt(name, array)

    mem_path = save_mem(name, array)

    c_path = save_c_header(name, array)

    # 保存参数元数据
    info = {
        "name": name,
        "shape": list(array.shape),
        "dtype": "float32",
        "flatten_order": "C",
        "statistics": get_array_statistics(array),
        "files": {
            "npy": str(
                npy_path.relative_to(ROOT)
            ).replace("\\", "/"),
            "txt": str(
                txt_path.relative_to(ROOT)
            ).replace("\\", "/"),
            "mem": str(
                mem_path.relative_to(ROOT)
            ).replace("\\", "/"),
            "c_header": str(
                c_path.relative_to(ROOT)
            ).replace("\\", "/")
        }
    }

    return info


# 验证导出文件
def verify_export(name, original_tensor):

    original = (
        original_tensor.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # 验证 NPY
    npy_path = NPY_DIR / f"{name}.npy"

    restored_npy = np.load(npy_path)

    if not np.array_equal(
        original,
        restored_npy
    ):
        raise AssertionError(
            f"NPY verification failed: {name}"
        )

    # 验证 MEM
    mem_path = MEM_DIR / f"{name}.mem"

    with open(
        mem_path,
        "r",
        encoding="ascii"
    ) as f:

        hex_values = [
            int(line.strip(), 16)
            for line in f
            if line.strip()
        ]

    restored_bits = np.array(
        hex_values,
        dtype=np.uint32
    )

    restored_mem = (
        restored_bits.view(np.float32)
        .reshape(original.shape)
    )

    if not np.array_equal(
        original,
        restored_mem
    ):
        raise AssertionError(
            f"MEM verification failed: {name}"
        )

    # 验证 TXT
    txt_path = TXT_DIR / f"{name}.txt"

    restored_txt = np.loadtxt(
        txt_path,
        dtype=np.float32
    ).reshape(original.shape)

    if not np.array_equal(
        original,
        restored_txt
    ):
        raise AssertionError(
            f"TXT verification failed: {name}"
        )

    print(f"  [PASS] {name}")


# 保存整体参数清单
def save_manifest(
    parameter_infos,
    total_params
):

    manifest = {
        "model": "LeNet",
        "precision": "FP32",
        "source_model": str(
            MODEL_PATH.relative_to(ROOT)
        ).replace("\\", "/"),
        "total_parameters": total_params,
        "total_weight_bytes": total_params * 4,
        "flatten_order": "C",
        "parameter_layout": {
            "conv_weight": [
                "out_channels",
                "in_channels",
                "kernel_height",
                "kernel_width"
            ],
            "fc_weight": [
                "out_features",
                "in_features"
            ],
            "bias": [
                "output_features"
            ]
        },
        "notes": [
            "Weights are exported in PyTorch native order.",
            "All arrays use C-order flattening.",
            "MEM files contain IEEE754 FP32 bit patterns.",
            "MEM files are not INT8 quantized weights.",
            "INT8 quantization will be handled separately."
        ],
        "parameters": parameter_infos
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=4,
            ensure_ascii=False
        )


def main():

    model = load_trained_model()

    state_dict = model.state_dict()

    parameter_infos = []

    total_params = 0

    print("\nExporting parameters...")
    print("-" * 65)

    for layer_name in LAYER_NAMES:

        for param_type in ["weight", "bias"]:

            name = (
                f"{layer_name}.{param_type}"
            )

            if name not in state_dict:

                raise KeyError(
                    f"Parameter not found: {name}"
                )

            tensor = state_dict[name]

            # 文件名中使用下划线
            file_name = name.replace(".", "_")

            info = export_parameter(
                file_name,
                tensor
            )

            parameter_infos.append(info)

            count = tensor.numel()

            total_params += count

            print(
                f"{file_name:<20} "
                f"Shape: {str(tuple(tensor.shape)):<20} "
                f"Count: {count}"
            )

    # 保存元数据
    save_manifest(
        parameter_infos,
        total_params
    )

    # 验证所有文件
    print("\nVerifying exported parameters...")
    print("-" * 65)

    for layer_name in LAYER_NAMES:

        for param_type in ["weight", "bias"]:

            name = (
                f"{layer_name}.{param_type}"
            )

            file_name = name.replace(".", "_")

            verify_export(
                file_name,
                state_dict[name]
            )

    expected_params = sum(
        p.numel()
        for p in model.parameters()
    )

    if total_params != expected_params:

        raise AssertionError(
            "Exported parameter count does not "
            "match model parameter count."
        )

    print("\n" + "=" * 65)
    print("Export Summary")
    print("=" * 65)

    print(f"Total Parameters : {total_params}")

    print(
        f"FP32 Storage     : "
        f"{total_params * 4} bytes"
    )

    print(
        f"FP32 Storage     : "
        f"{total_params * 4 / 1024:.2f} KiB"
    )

    print(f"Output Directory : {EXPORT_DIR}")

    print("\nAll weights exported successfully!")


if __name__ == "__main__":
    main()
