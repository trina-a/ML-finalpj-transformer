# ML-finalpj-transformer
modeling.py中为ViT架构  
train.py为配置参数和模型训练，训练采用baseline，mixup，cutout，cutmix四种方法  
load_cifar100.py为加载和划分cifar100数据集  
## 数据加载
train.py import load_cifar100.py加载和划分cifar100数据集，返回train_loader,test_loader，如果data文件夹中已有cifar100数据集，则直接加载，否则程序先进行下载
## 模型训练
将项目克隆到本地，并创建data和output两个文件夹，运行train.py即开始加载数据和训练模型，cifar100数据集会下载至data文件夹，分别训练出基于的baseline和经过mixup、cutout、cutmix后得出的模型以及loss、accuracy数据会输出到output文件夹中
## 训练出模型下载链接：
链接：https://pan.baidu.com/s/1I2Y6ROgc-UfoRKMX8ym8vA 
提取码：lo4k 
