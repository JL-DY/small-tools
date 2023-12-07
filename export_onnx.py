import torch
import torch.nn as nn
import onnx
import numpy as np

from utils import utils_deblur
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from models.network_usrnet import USRNet as net


# set your own kernel width
# kernel_width = 2.2
# kernel_width = 2
# k = utils_deblur.fspecial('gaussian', 25, kernel_width)
# k = sr.shift_pixel(k, 4)  # shift the kernel
# k /= np.sum(k)

# kernel = util.single2tensor4(k[..., np.newaxis])

# ----------------------------------------
# load model
# ----------------------------------------

model = net(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
            nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")


model.load_state_dict(torch.load("./model_zoo/usrnet_tiny.pth"), strict=True)
model.eval() # 若存在batchnorm、dropout层则一定要eval()!!!!再export
model = model.to("cpu")

# 输入参数
input=torch.randn((1,3,128,160))
kernel = torch.randn((1,1,25,25))
sigma = torch.tensor(2/255).float().view([1, 1, 1, 1])
sf = torch.tensor(4)

torch.onnx.export(model, (input, kernel, sf, sigma), './usrnet.onnx', input_names=["input", "kernel", "sf", "sigma"], output_names=["output"], export_params=True,opset_version=12,
                  dynamic_axes={'input': {2: 'height', 3: 'width'}, 'kernel': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}} )

print("Model has benn converted to onnx")
