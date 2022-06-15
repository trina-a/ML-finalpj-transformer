# ML-finalpj-transformer
data文件夹中存放cifar100数据集，可以通过load_cifar100.py下载，本项目中未上传  
output中存放输出的模型和loss、accuracy数据，本项目中未上传  
modeling.py中为ViT架构  
train.py为配置参数和模型训练，训练采用baseline，mixup，cutout，cutmix四种方法  
load_cifar100.py为加载和划分cifar100数据集  
## 数据加载
train.py import load_cifar100.py加载和划分cifar100数据集，返回train_loader,test_loader，如果data文件夹中已有cifar100数据集，则直接加载，否则程序先进行下载
## 模型训练
将项目克隆到本地，运行train.py即开始加载数据和训练模型，分别训练出baseline和经过mixup、cutout、cutmix后得出的模型，模型和loss、accuracy数据会输出到output文件夹中
## 训练出模型下载链接：
