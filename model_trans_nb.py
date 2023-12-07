from paddlelite.lite import *
# import paddlelite.lite as lite

model_path = r"E:\Project-paddle\PaddleSeg\JL-Project\MobileSeg\GhostNet\model_deploy"
save_path = r"C:\Users\lenovo\Desktop\all_nb_model\opencl_arm_model-naive_buffer-ghostnet"
# 1. 创建opt实例
# opt=lite.Opt()
opt = Opt()
# 2. 指定输入模型地址,combined形式
opt.set_model_dir(model_path)
#opt.set_model_file("E:/Project-paddle/PaddleSeg/JL-Project/MobileSeg/GhostNet/model_deploy/model.pdmodel")
#opt.set_model_file("E:/Project-paddle/PaddleSeg/JL-Project/MobileSeg/GhostNet/model_deploy/model.paiparams")
# 3. 指定转化类型： arm、x86、opencl、npu
opt.set_valid_places("opencl,arm") # 中间不能有空格
# 4. 指定模型转化类型： naive_buffer、protobuf
opt.set_model_type("naive_buffer")
# 4. 输出模型地址
opt.set_optimize_out(save_path)
# 5. 执行模型优化
opt.run()
