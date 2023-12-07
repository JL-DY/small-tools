import os
import numpy as np
import cv2 as cv
from PIL import Image


root_path = r"E:\Project-python\small_tools\visual"
root_path1 = r"E:\Project-python\small_tools\label"

lb_img_name = os.listdir(root_path)
for i, name in enumerate(lb_img_name):
    img = Image.open(os.path.join(root_path, name)).convert('P')
    img = np.array(img, dtype=np.uint8)
    img[img>0] = 1
    save_img = Image.fromarray(img, "P")
    save_img.save(os.path.join(root_path1, name))