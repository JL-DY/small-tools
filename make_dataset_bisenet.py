import os
import cv2 as cv
import numpy as np
import random


# 写入annotation
def write_annotations_to_txt(file_path, all_name, system):
    annotations = []

    if system not in ["windows", "linux"]:
        assert print("system名称只能为linux或windows,请重新输入")
    
    if system == "linux":
        for i in all_name:
            img_name = i[0]
            label_name= i[1]
            img_path = f"images/{img_name}"
            label_path = f"labels/{label_name}"
            annotations.append(img_path + ' ' + label_path)
    if system =="windows":
        for i in all_name:
            img_name = i[0]
            label_name= i[1]
            img_path = f"images\{img_name}"
            label_path = f"labels\{label_name}"
            annotations.append(img_path + ' ' + label_path)
    with open(file_path, 'w') as file:
        for annotation in annotations:
            file.write(annotation + '\n')


def extract_img_name(img_path, label_path):
    img_name = os.listdir(img_path)
    label_name = os.listdir(label_path)
    ratio = 0.2
    if len(img_name) == len(label_name): 
        name_sum = list(zip(img_name, label_name))
        print("图像和标签数量相同！")
        random.seed(510)
        random.shuffle(name_sum)
        # img_name[:], label_name[:] = zip(*name_sum)

        # num = int(ratio*len(img_name))

        # val_img_name = img_name[:num]
        # train_img_name = img_name[num:]

        # val_label_name = label_name[:num]
        # train_label_name = label_name[num:]
        num = int(ratio*len(img_name))
        train_name = name_sum[num:]
        val_name = name_sum[:num]
        print("train图像共分为{}张,val图像共分为{}张".format(len(train_name), len(val_name)))
        # return train_img_name, val_img_name, train_label_name, val_label_name
        return train_name, val_name
    else:
        print("图像和标签数量不相同,写入失败！")
        return []


"""
制作BiseNet训练所需的train.txt和val.txt
训练集和验证集打乱后按照8:2进行分开
"""
if __name__ == "__main__":
    # train.txt和val.txt文件路径
    file_train__path = r"C:\Users\lenovo\Desktop\custom-dataset\train.txt" # train.txt路径
    file_val__path = r"C:\Users\lenovo\Desktop\custom-dataset\val.txt"     # val.txt路径
    # images和labels目录路径
    img_path = r"C:\Users\lenovo\Desktop\custom-dataset\images" # images路径
    label_path = r"C:\Users\lenovo\Desktop\custom-dataset\labels"   # labels路径
    # 程序运行系统
    system = 'linux'


    # 开始区分数据集
    # train_img_name, val_img_name, train_label_name, val_label_name= extract_img_name(img_path, label_path)
    train_name, val_name = extract_img_name(img_path, label_path)
    # 写入train.txt和val.txt
    write_annotations_to_txt(file_train__path, train_name, system) # 写入train.txt
    write_annotations_to_txt(file_val__path, val_name, system)     # 写入val.txt
    print(f'注释已写入 {file_train__path}')
    print(f'注释已写入 {file_val__path}')

