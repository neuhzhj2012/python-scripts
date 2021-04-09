#encoding: utf-8
"""
源项目地址：https://github.com/ultralytics/yolov5
"""
import io
import json
import time
import requests
import os
import cv2
from PIL import Image
import argparse


import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

#模型label映射表
NAMES={'0':'二维码','1':'抖音','2':'皮皮虾','3':'快手'}

class YOLOV5():
    def __init__(self, weight_path="yolov5m.pt", img_size = 640, conf_thresh = 0.25, iou_thresh = 0.45):
        self.conf_thres = conf_thresh
        self.iou_thres = iou_thresh
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weight_path, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.augment = False

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        pass

    def predict(self, image_cv):
        # Padded resize
        img = letterbox(image_cv, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None,
                                   agnostic=False)[0]

        gn = torch.tensor(image_cv.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        result = dict()
        if len(det):
            # Rescale boxes from img_size to im0 size
            #print(img.shape[2:], det[:,:4], image_cv.shape)
            #print(img.shape[2:], det, image_cv.shape)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_cv.shape).round()

            # Write results
            det = sorted(det, key=lambda x: x[4], reverse=True)  # 根据置信度降序排列

            for *xyxy, conf, cls in reversed(det):
                label = NAMES[self.names[int(cls)]]
                conf = float(conf)
                xyxy = list(torch.tensor(xyxy).view(1, 4)[0])
                info = "{}_{}".format(label, conf)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                if label not in result:
                    result[label]=list()
                result[label].append([x1,y1,x2,y2,conf])

        return result

def show(img_cv, result):
    inv_map = {v: k for k, v in NAMES.items()}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in inv_map.keys()]
    for label_name, boxes in result.items():
        label_num = inv_map[label_name]
        color = colors[list(NAMES.keys()).index(label_num)]
        for box in boxes:
            x1, y1, x2, y2, conf = box
            info = "{}_{}".format(label_num, conf)

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color=color, thickness=3)
            cv2.putText(img_cv, info, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 2, color=color,
                        thickness=2)
    return img_cv


def getRszImg(img_cv, edge_long = 350):
    img_h, img_w = img_cv.shape[:2]
    ratio = img_h * 1.0 / img_w
    if ratio >=1.0:
        img_rsz_h = edge_long
        img_rsz_w = int(img_w * img_rsz_h * 1.0 / img_h)
    else:
        img_rsz_w = edge_long
        img_rsz_h = int(img_h * img_rsz_w * 1.0 / img_w)
    img_rsz = cv2.resize(img_cv, (img_rsz_w, img_rsz_h))
    return img_rsz

def getImgs(img_folder):
    rst = list()
    for sub_dir, _, names in os.walk(img_folder):
        for name in names:
            if name.split('.')[-1] not in ['jpg', 'jpeg', 'png']:
                continue
            rst.append(os.path.join(sub_dir, name))
    return rst

def getArgs():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', help='input img or folder or url',
                      default='img/1.jpg')
    return vars(args.parse_args())

if __name__ == '__main__':
    args = getArgs()
    in_file = args['input']

    if os.path.isfile(in_file): #可以为jpg,url
      if in_file[-3:]!='txt':
          img_paths = [in_file]
      else:
          print('input: {}'.format(in_file))
          buffs=open(in_file,'r').readlines()
          img_paths=[line.strip() for line in buffs]
    else:
      img_paths = getImgs(in_file)

    yolov5 = YOLOV5(weight_path="models/qrcode_watermark_v5m_022210.pt", img_size = 640, conf_thresh = 0.25, iou_thresh = 0.45)


    for img_path in img_paths:
        img_name = os.path.basename(img_path)

        img_cv = cv2.imread(img_path)
       
        tm_start = time.time()
        rst = yolov5.predict(img_cv)
        tm_pred = time.time() - tm_start

        print ('name:{}\ttm:{}\trst:{}'.format(img_name,tm_pred, rst))
        img_show = show(img_cv, rst)
        #cv2.imwrite('rst_'+img_name, img_show)


