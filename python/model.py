
"""
model.py

项目：基于 RISC-V SoC 的 LeNet 手写数字识别硬件加速器

功能：
1. 定义 LeNet 神经网络结构
2. 实现前向传播
3. 支持提取各层中间计算结果
4. 提供模型结构与参数量检查

输入：
    MNIST 灰度图片
    Shape: [N, 1, 28, 28]

输出：
    10个类别的原始得分（logits）
    Shape: [N, 10]

说明：
    使用适合硬件实现的 LeNet 变体。
    不使用 BatchNorm、Dropout 等额外操作。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        
        # 1. 卷积层
        # Conv1
        # 输入:  [N, 1, 28, 28]
        # 输出:  [N, 6, 24, 24]
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )

        # Conv2
        # 输入:  [N, 6, 12, 12]
        # 输出:  [N, 16, 8, 8]
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )

        # 2. 池化层
        # 2x2 最大池化
        # stride = 2
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 3. 全连接层
        # 输入: 16 x 4 x 4 = 256
        # 输出: 120
        self.fc1 = nn.Linear(
            in_features=256,
            out_features=120,
            bias=True
        )

        # 输入: 120
        # 输出: 84
        self.fc2 = nn.Linear(
            in_features=120,
            out_features=84,
            bias=True
        )

        # 输入: 84
        # 输出: 10
        self.fc3 = nn.Linear(
            in_features=84,
            out_features=10,
            bias=True
        )

    # 4. 标准前向传播
    def forward(self, x):

        # Conv1 -> ReLU -> Pool1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv2 -> ReLU -> Pool2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # FC1 -> ReLU
        x = self.fc1(x)
        x = F.relu(x)

        # FC2 -> ReLU
        x = self.fc2(x)
        x = F.relu(x)

        # FC3
        x = self.fc3(x)

        # 返回原始分类得分
        return x

    # 5. 提取中间层结果
    @torch.no_grad()
    def forward_with_intermediates(self, x):
        """
        返回各层输出，用于：
        1. Python与Verilog结果对比
        2. INT8量化分析
        3. 硬件模块功能验证
        """

        outputs = {}

        # Conv1
        x = self.conv1(x)
        outputs["conv1"] = x.detach().clone()

        # ReLU1
        x = F.relu(x)
        outputs["relu1"] = x.detach().clone()

        # Pool1
        x = self.pool(x)
        outputs["pool1"] = x.detach().clone()

        # Conv2
        x = self.conv2(x)
        outputs["conv2"] = x.detach().clone()

        # ReLU2
        x = F.relu(x)
        outputs["relu2"] = x.detach().clone()

        # Pool2
        x = self.pool(x)
        outputs["pool2"] = x.detach().clone()

        # Flatten
        x = torch.flatten(x, start_dim=1)
        outputs["flatten"] = x.detach().clone()

        # FC1
        x = self.fc1(x)
        outputs["fc1"] = x.detach().clone()

        # ReLU3
        x = F.relu(x)
        outputs["relu3"] = x.detach().clone()

        # FC2
        x = self.fc2(x)
        outputs["fc2"] = x.detach().clone()

        # ReLU4
        x = F.relu(x)
        outputs["relu4"] = x.detach().clone()

        # FC3
        x = self.fc3(x)
        outputs["fc3"] = x.detach().clone()

        return outputs


# 6. 模型测试
def test_model():

    print("=" * 60)
    print("LeNet Model Test")
    print("=" * 60)

    # 创建模型
    model = LeNet()
    model.eval()

    # 构造模拟MNIST输入
    # Batch Size = 1
    dummy_input = torch.randn(1, 1, 28, 28)

    # 前向传播
    with torch.no_grad():
        output = model(dummy_input)

    print("\nModel Architecture:")
    print(model)

    print("\nInput Shape:")
    print(tuple(dummy_input.shape))

    print("\nOutput Shape:")
    print(tuple(output.shape))

    # 参数总量
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("\nTotal Parameters:")
    print(total_params)

    # 检查中间层
    intermediate_outputs = (
        model.forward_with_intermediates(dummy_input)
    )

    print("\nIntermediate Layer Shapes:")

    for name, tensor in intermediate_outputs.items():
        print(f"{name:12s}: {tuple(tensor.shape)}")

    # 检查输出形状
    assert output.shape == (1, 10), \
        "Output shape is incorrect!"

    # 检查两种前向传播结果一致
    assert torch.allclose(
        output,
        intermediate_outputs["fc3"]
    ), "Forward outputs do not match!"

    print("\nAll model tests passed!")


if __name__ == "__main__":
    test_model()
