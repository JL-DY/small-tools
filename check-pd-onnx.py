# 导入所需的库
import numpy as np
import onnxruntime
import paddle

# 随机生成输入，用于验证 Paddle 和 ONNX 的推理结果是否一致
x = np.random.random((1, 3, 256, 256)).astype('float32')

# predict by ONNXRuntime
ort_sess = onnxruntime.InferenceSession("E:/Project-paddle/PaddleSeg/JL-Project/MobileSeg/GhostNet/onnx/ghostnet.onnx")
ort_inputs = {ort_sess.get_inputs()[0].name: x}
ort_outs = ort_sess.run(None, ort_inputs)

print("Exported model has been predicted by ONNXRuntime!")

# predict by Paddle
model = paddle.jit.load("E:/Project-paddle/PaddleSeg/JL-Project/MobileSeg/GhostNet/model_deploy-none/model")
model.eval()
paddle_input = paddle.to_tensor(x)
paddle_outs = model(paddle_input)

print("Original model has been predicted by Paddle!")

# compare ONNXRuntime and Paddle results
np.testing.assert_allclose(ort_outs[0], paddle_outs.numpy(), rtol=1.0, atol=1e-05)

print("The difference of results between ONNXRuntime and Paddle looks good!")

