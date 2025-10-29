# Kaggle笔记

## 1.下载数据集

1.除了使用download按钮下载数据集外，使用python脚本也可下载，首先安装需要的两个包

```bash
pip install kaggle kagglehub -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2.安装好后使用download.py下载数据集，以CIFAR-100举例

```python
import kagglehub
import shutil
import os

# 1. 下载（到默认缓存目录）
path = kagglehub.dataset_download("fedesoriano/cifar100")
print("默认下载路径:", path)

# 2. 设定你想要的目标路径
target_dir = "./mydata/cifar100"  # ✅ 自定义目录

# 如果目录不存在就创建
os.makedirs(target_dir, exist_ok=True)

# 3. 移动文件
if os.path.exists(path):
    shutil.move(path, target_dir)
    print("已移动到:", target_dir)
else:
    print("下载路径不存在:", path)

```

```
Warning: Looks like you're using an outdated `kagglehub` version, please consider updating (latest version: 0.3.13)
默认下载路径: /home/wjj3090/.cache/kagglehub/datasets/fedesoriano/cifar100/versions/1
已移动到: ./mydata/cifar100
```

3.之后处理下载好的数据集（CIFAR-100）：

```python
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

metadata_path = './mydata/cifar100/1/meta' # 读取元数据路径
metadata = unpickle(metadata_path)
superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))

data_pre_path = './mydata/cifar100/1/' # change this path
# File paths
data_train_path = data_pre_path + 'train'
data_test_path = data_pre_path + 'test'
# Read dictionary
data_train_dict = unpickle(data_train_path)
data_test_dict = unpickle(data_test_path)
# Get data (change the coarse_labels if you want to use the 100 classes)
data_train = data_train_dict[b'data']
train_labels = np.array(data_train_dict[b'coarse_labels'])
data_test = data_test_dict[b'data']
test_labels = np.array(data_test_dict[b'coarse_labels'])

# 将每张图片从 3072 向量还原为 32x32x3
train_images = data_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
test_images = data_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
# transpose(0, 2, 3, 1)调整维度顺序：原始维度索引：0（样本数）、1（通道）、2（高度）、3（宽度）调整后维度索引变为 0（样本数）、2（高度）、3（宽度）、1（通道）最终形状为 (样本数, 32, 32, 3)，即 (N, H, W, C) 格式（通道在后，常见于 TensorFlow 或 matplotlib 显示图像时的格式）

print("训练集图像形状：", train_images.shape)  # (50000, 32, 32, 3)
print("测试集图像形状：", test_images.shape)    # (10000, 32, 32, 3)
print("训练集标签形状：", train_labels.shape)
print("测试集标签形状：", test_labels.shape)

import matplotlib.pyplot as plt

idx = 1  # 你想看的图像编号
plt.imshow(train_images[idx])
plt.axis('off')
plt.show()
```

<img src="../assests/Kaggle笔记/image-20251015112555952.png" alt="image-20251015112555952" style="zoom:33%;" />

