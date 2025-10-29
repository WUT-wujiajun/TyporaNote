# Graspnet数据集环境配置：

服务器配完环境后最后没法复现，因为没有图形化显示，建议使用本地双系统使用ubuntu复现



2080ti服务器系统cuda版本10.1，python用3.8（3.7安装graspnetAPI会报错）

本机cuda版本也是10.1



提前安装sudo apt install nvidia-cuda-toolkit查看系统cuda版本



## 清华源

```bash
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 1.创建新环境

```
conda create -n graspnet python=3.8

conda activate graspnet
```

## 2.安装与系统cuda对应版本的torch

pytorch历史版本：https://pytorch.org/get-started/previous-versions/

nvcc -V显示出来的cuda版本是多少，就去下载对应cuda版本的pytorch，使用1.7.1老版本pytorch可以避免一些报错

因为本机和服务器版本一样，所以都使用：

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
或者
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## 3.下载源码并安装依赖

```
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
```

在requirement.txt中把torch用#注释掉，然后安装

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4.安装pointnet2

```
cd pointnet2
python setup.py install
```

## 5.安装knn

这一步需要修改 knn/src/knn.h

在文件开头添加：

```c++
#include <c10/cuda/CUDACachingAllocator.h>
```

然后找到这两行：

```c++
float *dist_dev = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(ref_nb * query_nb * sizeof(float));
c10::cuda::CUDACachingAllocator::raw_delete(dist_dev);
```

替换为：

```c++
float *dist_dev = static_cast<float*>(c10::cuda::CUDACachingAllocator::raw_alloc(ref_nb * query_nb * sizeof(float)));
c10::cuda::CUDACachingAllocator::raw_delete(dist_dev);
```

替换之后终端执行：

```
cd knn
python setup.py install
```

## 6.安装graspnetAPI

需要先将setup.py中的 “sklearn”改为 “scikit-learn”

```
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 7.测试

把checkpoint下载下来

https://github.com/graspnet/graspnet-baseline?tab=readme-ov-file

checkpoint-kn.tar 

在graspnet-baseline下创建文件夹logs/log_kn

```
cd graspnet-baseline
mkdir -p logs/log_kn
```

把下好的checkpoint-kn.tar重命名为checkpoint.tar，放到里面。后运行

```
sh command_demo.sh
```



