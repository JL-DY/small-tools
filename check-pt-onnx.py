import torch
import numpy as np


def pytorch_out(input):
    model = torch.load("C:/Users/lenovo/Desktop/model_DnCNN_datav1_epoch_500.pth")["model"] #model.eval
    # input = input.cuda()
    # model.cuda()
    torch.no_grad()
    model.eval()
    output = model(input)
    # print output[0].flatten()[70:80]
    return output

def pytorch_onnx_test():
    import onnxruntime
    from onnxruntime.datasets import get_example

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # 测试数据
    torch.manual_seed(66)
    dummy_input = torch.randn(1, 360, 480, device='cpu')

    example_model = get_example("C:/Users/lenovo/Desktop/dncnn-sigma11.onnx")
    # netron.start(example_model) 使用 netron python 包可视化网络
    sess = onnxruntime.InferenceSession(example_model)

    # onnx 网络输出
    onnx_out = np.array(sess.run(None, { "inputs": to_numpy(dummy_input)}))  #fc 输出是三维列表
    print("==============>")
    print(onnx_out)
    print(onnx_out.shape)
    print("==============>")
    torch_out_res = pytorch_out(dummy_input).detach().numpy()   #fc输出是二维 列表
    print(torch_out_res)
    print(torch_out_res.shape)

    print("===================================>")
    print("输出结果验证小数点后五位是否正确,都变成一维np")

    torch_out_res = torch_out_res.flatten()
    onnx_out = onnx_out.flatten()

    pytor = np.array(torch_out_res,dtype="float32") #need to float32
    onn=np.array(onnx_out,dtype="float32")  ##need to float32
    np.testing.assert_almost_equal(pytor,onn, decimal=5)  #精确到小数点后5位，验证是否正确，不正确会自动打印信息
    print("恭喜你 ^^, onnx 和 pytorch 结果一致, Exported model has been executed decimal=5 and the result looks good!")

pytorch_onnx_test()