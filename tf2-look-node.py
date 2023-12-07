import tensorflow as tf

model_dir = "C:/Users/lenovo/Desktop/ESPCN_x4.pb"

# 读取模型
with tf.io.gfile.GFile(model_dir, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()  # 使用 tf.compat.v1.GraphDef() 支持 TensorFlow 1.x GraphDef
    graph_def.ParseFromString(f.read())

    # 打印输入节点的结构
    print('>>>打印输入节点的结构如下：\n', graph_def.node[0])
    print('*' * 50, '\n')

    # 打印输出节点的结构
    print('>>>打印输出节点的结构如下：\n', graph_def.node[-1])
