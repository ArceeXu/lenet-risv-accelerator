
from pathlib import Path
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
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

print("训练集数量:", len(train_dataset))
print("测试集数量:", len(test_dataset))

image, label = train_dataset[0]

print("图片形状:", image.shape)
print("标签:", label)
print("像素范围:", image.min().item(), image.max().item())
print("数据集路径:", DATA_DIR)
