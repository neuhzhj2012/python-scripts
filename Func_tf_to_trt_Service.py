#encoding: utf-8
'''
镜像：nvcr.io/nvidia/tensorrt:18.06-py3
参考文件：tensorrt/python/examples/tf_to_trt/tf_to_trt.py
uff模块下载tensor安装包(对应到具体版本)
trt模型保存与重载参考https://mp.weixin.qq.com/s/Ps49ZTfJprcOYrc6xo-gLg?
https://docs.nvidia.com/deeplearning/sdk/tensorrt-release-notes/index.html
https://developer.nvidia.com/nvidia-tensorrt-download
'''
from __future__ import division
from __future__ import print_function
import os
import cv2
import argparse
import time
import numpy as np

#import uff #该模块影响logging文件输出
try:
    import tensorrt as trt
    from tensorrt.lite import Engine
    from tensorrt.parsers import uffparser
except:
    import tensorrt.legacy as trt
    from tensorrt.legacy.lite import Engine
    from tensorrt.legacy.parsers import uffparser  #trt5

def parseArgs():
    args = argparse.ArgumentParser()
    args.add_argument("-i", '--image', help="input img or folder")
    return vars(args.parse_args())

def getAllFiles(folder):
    rtn = list()
    for dirname, _, imgs in os.walk(folder):
        for img in imgs:
            rtn.append(os.path.join(dirname, img))
    return rtn

def centralCropByNp(img_np, central_fraction=0.875):
    img_h, img_w = img_np.shape[:2]
    y_start = int(img_h * (1 - central_fraction) / 2)
    x_start = int(img_w * (1 - central_fraction) / 2)
    y_end = img_h - y_start * 2
    x_end = img_w - x_start * 2
    return img_np[y_start:y_start + y_end, x_start:x_start + x_end, :]

def preprocess_image(img):
    img = centralCropByNp(img, central_fraction=0.875)
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img = cv2.resize(img, (299, 299))
    img = 2 * (img / 255.) - 1
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, 0)
    return img

def getLabels(label_path='labels.txt'):
    label_map = {}
    label_map[0] = 'background'
    f = open(label_path, 'r')
    for index, line in enumerate(f.readlines()):
        label_map[index + 1] = line.strip()
    f.close()
    return label_map

def getTop5(confs, labels):
    result = list()
    confs_one = confs[0][0][0][0]
    top5_idxs = np.argsort(-np.array(confs_one))[:5]
    for idx in (top5_idxs).tolist():
        label = labels[idx]
        score = confs_one[idx]
        result.append((label, score))
    return result

def main():
    trt_engine_model = 'car_series/car_series_tensorrt.engine'

    tm_start=time.time()
    engine = Engine(PLAN=trt_engine_model)
    tm_load = time.time() - tm_start
    print ('tm_load: {}'.format(tm_load))
    assert(engine)

    args = parseArgs()
    imgs = args['image']
    if os.path.isfile(imgs):
        imgs_path=[imgs]
    else:
        imgs_path = getAllFiles(imgs)

    #warmup
    for _ in range(3):
        img = cv2.imread(imgs_path[0])
        img = preprocess_image(img)
        out = engine.infer(img)

    idx_label_dict = getLabels('car_series/labels.txt')
    tm_sum = 0
    for img_path in imgs_path:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        tm_start = time.time()
        img=preprocess_image(img)
        out = engine.infer(img)
        rst = getTop5(out, idx_label_dict)
        tm_svc = time.time() - tm_start
        tm_sum += tm_svc
        print ('{} rst: {}, tm: {}'.format(img_name, rst, tm_svc))
    print ('tm_ave: {}, num: {}'.format(tm_sum/len(imgs_path), len(imgs_path)))

if __name__ == "__main__":
    main()
