# image-algorithms
image-algorithms


基础模型：Unet-resnet50
模型框架：pytorch_lightning
数据集处理：
train_set与valid_set使用同样的初始化方式，init将数据集样本路径读取到list
getitem在训练或者验证期间读取增广数据，并对数据进行normalize；然后对label做处理，将label中的灰度值转换为标签值，eg，0,128,255转为0,2,1
训练：
对模型进行微调，根据训练经验一般训练100~500epoch。
训练后对训练集的效果进行图像保存，进行初步检验。具体代码在image_seg_dl_predict.py脚本里面

测试：
测试分别采用可视化边缘，图像分割评估指标以及分布图方法。具体代码在image_seg_dl_predict.py
Visualize, evalueate函数。

对外可调参数：batchsize，epoch， num_class
