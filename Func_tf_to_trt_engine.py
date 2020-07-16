#encoding: utf-8
'''
镜像：nvcr.io/nvidia/tensorrt:18.06-py3
参考文件：tensorrt/python/examples/tf_to_trt/tf_to_trt.py
uff模块下载tensor安装包(对应到具体版本)
trt模型保存与重载参考https://mp.weixin.qq.com/s/Ps49ZTfJprcOYrc6xo-gLg?
https://developer.nvidia.com/tensorrt
https://developer.nvidia.com/nvidia-tensorrt-download
ckpt-tf1.2ok
'''
from __future__ import division
from __future__ import print_function
import uff
try:  #trt4
    import tensorrt as trt
    from tensorrt.parsers import uffparser
except:  #trt5
    import tensorrt.legacy as trt
    from tensorrt.legacy.parsers import uffparser  #trt5

MAX_WORKSPACE = 1 << 30
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.WARNING)

CHANNEL = 3
INPUT_W = 299
INPUT_H = 299

MAX_BATCHSIZE = 1

def main():
    tf_freeze_model = 'car_series/frozen_graph.pb'
    input_node = 'input'
    out_node = 'InceptionV4/Logits/Predictions'

    uff_model = uff.from_tensorflow_frozen_model(tf_freeze_model, [out_node])
    #Convert Tensorflow model to TensorRT model
    parser = uffparser.create_uff_parser()
    parser.register_input(input_node, (CHANNEL, INPUT_H, INPUT_W), 0)
    parser.register_output(out_node)

    engine = trt.utils.uff_to_trt_engine(G_LOGGER,
                                              uff_model,
                                              parser,
                                              MAX_BATCHSIZE,
                                              MAX_WORKSPACE)

    trt.utils.write_engine_to_file("car_series/car_series_tensorrt.engine", engine.serialize())


if __name__ == "__main__":
    main()
