# 课题3 Lenet 手写识别加速器代码说明

本项目使用 MNIST 训练 LeNet，将浮点模型转换为 INT8 整数模型，再用 C 复现推理，为后续 E203 SoC 与 Verilog 加速器提供参考。

目前已完成 Python 训练、量化和 PC 上的 C 整数推理。硬件设计、ICB 总线集成和上板性能验证仍待完成。题目原文见 [03_lenet_acc.pdf](03_lenet_acc.pdf)。

## 1. 当前结果

| 验证项目 | 已有结果 | 证据 |
| --- | --- | --- |
| Python FP32，10000 张 | 准确率 98.91% | `results/accuracy/evaluation_metrics.json` |
| Python INT8，10000 张 | 准确率 98.86% | `results/quantization/quantization_metrics.json` |
| C 完整网络，10 张 | 每张 8 个 INT8 检查点全部一致 | `results/c_validation/lenet_int8_10samples.log` |
| C 完整网络，10000 张 | 准确率 98.86%，与 Python INT8 预测一致率 100% | `results/c_validation/full_test_metrics.json` |

## 2. 主流程

```text
MNIST 图片和标签
  → train.py：训练 → lenet_fp32.pth
  → quantize.py：校准、量化、Python 整数推理
      ├─ data/weights/int8：权重、偏置、重新量化参数
      ├─ data/test_vectors/int8：少量图片的逐层参考数据
      └─ results/quantization：准确率、全测试集预测记录
  → C 完整网络：卷积 → 池化 → 卷积 → 池化 → 三层全连接
      ├─ test_lenet：逐层检查整数结果
      └─ evaluate_lenet：检查完整测试集准确率和预测一致率
  → 后续：RTL → ICB/E203 → FPGA 上板
```

训练决定模型参数；量化决定整数表示与计算规则；C 使用已导出的整数参数，不重新训练、不重新选择量化尺度。标签和 Python 预测只用于评价，不参与 C 分类计算。

## 3. 文件各自的功能

```text
python/        模型、训练、量化、参考计算与数据导出
software/      C 计算模块和两个验证入口
data/          原始数据、模型权重、参考向量、全测试集输入
results/       训练与评估结果、必要验证记录
rtl/、tb/      预留：Verilog 和仿真
soc/、fpga/    预留：SoC 集成与开发板工程
docs/          预留：设计、排错和 AI 使用记录
README.md      使用说明
```

### Python

| 文件 | 作用 |
| --- | --- |
| `model.py` | 定义网络结构，其他 Python 脚本复用 |
| `download_mnist.py` | 下载 MNIST，仓库里把MNIST上传了，不用重新运行下载 |
| `train.py` | 训练 FP32 模型，保存 checkpoint 与训练记录 |
| `evaluate.py` | FP32 准确率、混淆矩阵和参考样本导出 |
| `quantize.py` | 用训练样本校准，生成 INT8 参数，运行整数参考推理 |
| `export_c_eval.py` | 使用标准库整理完整测试集和已有 Python INT8 预测，不重新推理 |
| `export_weights.py` | 可选：导出 FP32 参数，供浮点原理验证使用 |
| `conv_reference.py` | 可选：NumPy 浮点计算与 PyTorch 逐层比较，保留作原理验证依据 |

FP32 参考路线与当前 INT8 C 路线不同，因此保留浮点原理验证工具；日常 C 测试不需要运行它们。

### C

| 文件 | 作用 |
| --- | --- |
| `conv_int8.c/.h` | 卷积与共用重新量化函数 |
| `pool_int8.c` | 独立的 2×2 最大池化 |
| `fc_int8.c` | 独立的全连接、偏置、重新量化和可选 ReLU |
| `lenet_int8.c/.h` | 完整网络调度、接口和两块工作缓冲区 |
| `test_lenet.c` | 读取少量向量，比较完整网络的各层 INT8 输出 |
| `evaluate_lenet.c` | 对 10000 张图片推理，比较真实标签和 Python 预测 |

两个验证入口分别有 main，必须分别编译。核心计算不依赖文件系统，读取文件和保存报告由验证入口处理。

### 数据与结果

| 路径 | 用途 |
| --- | --- |
| `data/raw/MNIST/raw/` | MNIST 原始 IDX 数据或 gzip 文件 |
| `data/checkpoints/lenet_fp32.pth` | 已训练模型 |
| `data/weights/int8/` | C 使用的参数；`.h` 编译入程序，`.mem` 可用于硬件仿真，`.npy` 用于 Python |
| `data/weights/fp32/` | 浮点参考参数，不是 INT8 C 网络的输入 |
| `data/test_vectors/int8/` | 默认 10 张图片的输入和逐层预期结果，包括累加参考值 |
| `data/test_vectors/` 中其他数据 | 浮点参考样本和历史中间结果 |
| `data/c_eval/` | 10000 张原始图片、标签和 Python INT8 预测 |
| `results/accuracy/` | FP32 评估、训练曲线与日志 |
| `results/quantization/` | Python INT8 评估；预测 CSV 是全测试集导出的必要输入，不可随意删除 |
| `results/conv_reference/` | 浮点原理验证证据 |
| `results/c_validation/` | 10 张逐层日志、全测试集指标 JSON 和逐样本 CSV |

## 4. 在 Ubuntu 验证 C（自用，我将本地文件夹与 Ubuntu 共享了，方便使用 GPT）

Windows 的 `D:\lenet-riscv-accelerator` 和 Ubuntu 的共享目录是同一份文件。所有命令从项目根目录执行；可执行程序放到 Ubuntu 本地，避免共享目录的执行权限问题。

```bash
cd /mnt/hgfs/lenet-riscv-accelerator
mkdir -p ~/lenet-build results/c_validation
```

Ubuntu 需要 GCC；数据导出需要 Python 3 标准库，无需 PyTorch。不要直接编译 `software/*.c`，否则两个 main 会冲突。

### A. 逐层验证，默认检查 10 张，用来检查代码修改后是否正确

```bash
gcc -std=c99 -O2 -Wall -Wextra -Werror \
    software/conv_int8.c software/pool_int8.c software/fc_int8.c \
    software/lenet_int8.c software/test_lenet.c \
    -o ~/lenet-build/test_lenet

~/lenet-build/test_lenet 1
~/lenet-build/test_lenet
```

不带参数默认检查 10 张，要求对应向量已存在。检查 relu1、pool1、relu2、pool2、flatten、relu3、relu4、fc3，所有中间输入均由 C 计算。成功时输出 `All full-network INT8 layer checks passed.`。

需要更新逐层实验日志时：

```bash
set -o pipefail
~/lenet-build/test_lenet 2>&1 | tee results/c_validation/lenet_int8_10samples.log
echo "Test exit status: $?" | tee -a results/c_validation/lenet_int8_10samples.log
```

最后一条须紧接测试执行，状态 0 表示成功。会覆盖同名日志，重要旧记录请先保留。

### B. 全测试集验证，用来确认准确率和Python的INT8预测是否一致

```bash
python3 python/export_c_eval.py

gcc -std=c99 -O2 -Wall -Wextra -Werror \
    software/conv_int8.c software/pool_int8.c software/fc_int8.c \
    software/lenet_int8.c software/evaluate_lenet.c \
    -o ~/lenet-build/evaluate_lenet

~/lenet-build/evaluate_lenet
echo "Evaluation exit status: $?"
```

程序自动保存 `full_test_metrics.json` 和 `full_test_predictions.csv` 到 `results/c_validation/`。不再额外保存重复的进度日志。

- 准确率：C 预测与真实标签的比较。
- 一致率：C 预测与 Python INT8 预测的比较。
- 退出状态 0：评估完成，预测全部一致；2：评估完成但有不一致；1：数据、计算或写入失败，此时报告可能不完整。

导出脚本检查 10000 条记录的顺序和标签，支持原始 IDX 和 gzip，保存文件哈希到 `data/c_eval/manifest.json`。它复用已有 Python 预测，不能证明旧预测与刚修改的权重属于同一次运行。量化参数变化时必须按下一节重新生成。

## 5. 何时需要重跑 Python

已有训练和量化结果时，不必重复训练。需要重建时，在安装了 PyTorch、torchvision、NumPy、matplotlib、tqdm 的训练环境中，从项目根目录按需运行：

```bash
python python/download_mnist.py
python python/train.py
python python/evaluate.py
python python/quantize.py --test-samples 10000
python python/export_c_eval.py
```

`quantize.py` 默认评估 100 张，完整评估必须指定 10000。默认校准 512 张训练图片、导出 10 张详细向量；可通过 `--calibration-samples` 和 `--export-vectors` 调整。

可选浮点原理验证：先运行 `python python/export_weights.py`，再运行 `python python/conv_reference.py`。

| 修改内容 | 要重做的步骤 |
| --- | --- |
| 只修改 C | 重新编译并运行两种 C 验证 |
| 修改量化规则或参数 | 完整 Python 量化评估 → 导出 C 评估数据 → 重新编译和验证 C |
| 重新训练 | 更新 FP32 评估、量化、C 数据与 C 可执行程序 |
| 改变网络结构 | 同步修改 C 尺寸、缓冲区和调度，再完成整条验证流程 |

## 6. 必须保持一致的计算规则

网络使用 28×28 输入，5×5 卷积核，步长 1，无填充；卷积不翻转核。特征布局 CHW、卷积权重 OIHW、全连接权重 OI，均按 C 顺序连续存储。

```text
输入 1×28×28
→ Conv1/ReLU 6×24×24 → Pool1 6×12×12
→ Conv2/ReLU 16×8×8 → Pool2 16×4×4
→ Flatten 256 → FC1/ReLU 120 → FC2/ReLU 84 → FC3 10 → Argmax
```

- 原像素转换：`(pixel * 127 + 127) / 255`，使用非负整数计算后转 INT8。
- 权重为逐输出通道对称 INT8，激活为逐张量对称 INT8，零点 0；本项目有符号范围为 -127..127。
- 偏置为 INT32，参数量化使用最近舍入、半数取偶。
- C 参考计算用 INT64 累加，检查最终 INT32 范围；硬件逐步累加的溢出边界仍需另行分析。
- 重新量化使用 INT64 乘积、逐输出 multiplier 和 shift（0..62），最近舍入且半数远离零，然后饱和。
- ReLU 输出 0..127；FC3 没有 ReLU，输出 -127..127。最大池化不改变量化尺度。
- FC3 得分同尺度，直接 Argmax；并列最大值取第一个。得分不是概率。

### 上述INT8、INT32 与 INT64 在量化推理中的简单解释（自用）

- **量化的基本思想**：用整数和缩放系数近似表示浮点数，即 `真实数值 ≈ 整数值 × scale`。例如 `scale = 0.01` 时，整数 `12` 表示约 `0.12`。这种转换会产生舍入误差，但能减少存储和计算资源。
- **输入、权重和层输出使用 INT8**：每个数占 1 字节，比 FP32 的 4 字节更省空间。本项目采用对称范围 `-127～127`，零点为 0；ReLU 后只保留 `0～127`。权重每个输出通道有独立尺度，激活每个张量使用统一尺度。
- **累加结果需要 INT32 范围**：两个 INT8 数相乘就可能达到 `127 × 127 = 16129`，卷积和全连接还需要累加多个乘积，因此不能继续用 INT8 保存结果。例如第二层卷积每个输出累加 150 个乘积，仅乘积和的最大绝对值估计就达到 `2,419,350`，尚未包含偏置。
- **偏置使用 INT32**：偏置直接加到累加结果上，必须使用与累加结果相同的尺度，即 `输入 scale × 权重 scale`。这一尺度通常较小，偏置转换后的整数可能较大，因此使用 INT32。
- **C 参考程序使用 INT64 累加以避免中间溢出**：累加完成后，再检查结果是否能放入 INT32。后续硬件若采用 INT32 累加器，还需要分析每一步的部分和是否越界，不能只检查最终结果。
- **重新量化使用 INT64 中间乘积**：将累加结果转换为下一层的 INT8 尺度时，使用 `输出整数 ≈ 累加整数 × multiplier / 2^shift`。乘法中间结果可能超过 INT32，因此暂用 INT64，随后进行舍入、移位和饱和，得到 INT8 输出。
- **舍入、饱和和 ReLU 各有作用**：舍入将小数结果变成整数；饱和将越界结果限制在允许范围内；ReLU 将负数变为零。本项目参数量化采用“最近舍入、半数取偶”，重新量化采用“最近舍入、半数远离零”，Python、C 和硬件必须遵循对应规则。
- **最大池化与分类输出**：最大池化只选取最大值，不改变量化尺度。FC3 不使用 ReLU，保留负数；其 10 个输出使用相同尺度，因此可直接比较整数大小，用 Argmax 得到类别，得分本身不是概率。
- **量化精度与计算位宽不同**：INT8 决定数据表示的精细程度；INT32／INT64 用于容纳较大的中间结果，避免溢出。扩大中间位宽不会恢复量化时丢失的信息，因此“INT8 模型”并不意味着所有运算都使用 8 位。

参数和详细规则见 `data/weights/int8/quantization_manifest.json`。FP32 `.mem` 是浮点位模式，不能替代 INT8 参数。

## 7. 内存和调用接口

`lenet_predict` 接收 784 个已量化 INT8 输入、调用者提供的 workspace、10 个 INT8 scores 和 prediction。两块特征缓冲区共 6912 字节，输入另需 784 字节，输出另需 10 字节；Flatten 不复制数据。

权重、偏置、multiplier、shift 共 46748 字节，不含对齐、栈和程序代码。const 的实际存放位置由目标链接脚本决定。PC 测试的参考数组和文件缓冲不属于部署必需内存。

缓冲区必须足够大且互不重叠，池化和 FC 不支持原地操作；并发调用使用独立 workspace。观察回调同步执行，不应保留会被复用的特征指针。部署时 observer/context 传 NULL。计算返回 0 表示成功，-1 表示参数无效，-2 表示累加越界，完整网络的 -3 表示观察回调失败；失败输出不可使用。

卷积接口限定当前网络规模：输入通道最多 6，输出通道最多 16，空间尺寸最多 28，卷积核为 5。卷积和 FC 的可选 accumulators 指针可用于检查量化前累加值，传 NULL 时不保存整层累加数组。

## 8. 后续工作

后续需要确定开发板资源，设计并验证 RTL、接入 E203 ICB 总线、完成串口收发和上板计时。题目要求主要设计与分析由小组完成，并保留实验、排错、版本和 AI 使用记录；不要把必要实验依据当临时日志清理。性能对比应使用 RISC-V 上的软件和同一硬件计时基准，不能使用 Ubuntu 耗时代替。

