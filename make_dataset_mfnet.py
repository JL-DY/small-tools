import os
import cv2 as cv
import numpy as np
import random

def extract_rgb_and_t(img_path):
    img_name = os.listdir(img_path)
    for i, name in enumerate(img_name):
        img_abs_path = os.path.join(img_path, name)
        img = np.array(cv.imread(img_abs_path, cv.IMREAD_UNCHANGED), dtype=np.uint8)
        bgr_img, t_img = img[:, :, :3], img[:, :, 3]
        cv.imwrite(os.path.join(save_rgb_img, name.replace(".png", ".jpg")), bgr_img) # 为了跑模型方便将img的png格式都保存为了jpg，可根据自己的需求调整
        cv.imwrite(os.path.join(save_t_img, name.replace(".png", ".jpg")), t_img)
    print('RGB图和温度图已提取完成')


def trans_label(label_path):
    label_name = os.listdir(label_path)
    for i, name in enumerate(label_name):
        img_abs_path = os.path.join(label_path, name)
        img = np.array(cv.imread(img_abs_path, cv.IMREAD_UNCHANGED), dtype=np.uint8)
        img[img>2] = 0
        img_copy = img.copy()
        img[img_copy==2] = 1
        img[img_copy==1] =0
        cv.imwrite(os.path.join(save_our_label, name), img)
    print('标签已转换为背景0、人1两类')


if __name__ == "__main__":
    pri_img_path = "E:/Project-python/DATA/MFnet/ir_seg_dataset/images"
    pri_label_path = "E:/Project-python/DATA/MFnet/ir_seg_dataset/labels"

    save_rgb_img = "E:/Project-python/DATA/MFnet/ir_seg_dataset/RGB_img"
    save_t_img = "E:/Project-python/DATA/MFnet/ir_seg_dataset/TEMP_img"
    save_our_label = "E:/Project-python/DATA/MFnet/ir_seg_dataset/our_labels"

    # extract_rgb_and_t(pri_img_path)
    trans_label(pri_label_path)
    print('所有操作已执行完毕')
