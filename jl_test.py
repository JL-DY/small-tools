import cv2 as cv
import numpy as np
import os

if __name__ == "__main__":
    input_path = "C:/Users/lenovo/Desktop/input-images"
    img_list = os.listdir(input_path)
    result_path = "C:/Users/lenovo/Desktop/input-resize"
    for i in img_list:
        img_file = os.path.join(input_path, i)
        img = cv.imread(img_file)

        w, h, c = img.shape
        img = cv.resize(img, (4*h, 4*w), cv.INTER_LANCZOS4)
        cv.imwrite(os.path.join(result_path, i), img)

