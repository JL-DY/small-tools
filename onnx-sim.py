import onnx
from onnxsim import simplify

onnx_path = "C:/Users/lenovo/Desktop/2023-1206-1337.onnx"
save_path = "C:/Users/lenovo/Desktop/ecbsr_dncnn-sim.onnx"

# 加载ONNX模型
model = onnx.load(onnx_path)

# 剪枝模型权重
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

# 保存剪枝后的模型
onnx.save(model_simp, save_path)
