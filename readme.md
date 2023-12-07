这个仓库主要是用来保存一些个人平时会用到的脚本或代码，以免每次都重新写或去网上找，在使用过程中会逐渐对一些代码进行相应的完善。


​	每个文件代码都是一个独立的功能，以下是它们的大致描述

------

check-pd-onnx.py：用来检查paddle训练的.pd模型和导出的onnx模型是否一致

check-pt-onnx.py：用来检查torch训练的.pt模型和导出的onnx模型是否一致

export_onnx.py：将torch模型导出为onnx

onnx-sim.py：对导出的onnx模型进行简化

Extract_image_frame.py：对视频进行抽帧，获取数据集

jl_test.py：写一些临时代码用的文件(可不看)

label_convert.py：针对分割数据集，对label标签数据的批量转换

lrh_code.cpp：边缘增强的c++代码(临时留着的)

make_dataset_bisenet.py：根据bisenet分割网络，对train.txt、val.txt文件编写的脚本

make_dataset_mfnet.py：根据mfnet红外数据集，对图像rgb图和t图进行分离，并实现相应的标签转换

model_trans_nb.py：将paddle训练并且export得到的两个模型文件.pdmodel和.paiparams，转换为paddlelite推理用的.nb模型文件

tf2-look-node.py：查看tf2版本保存的模型的节点名称(此代码有问题，建议直接用[netron](https://netron.app/)查看)
