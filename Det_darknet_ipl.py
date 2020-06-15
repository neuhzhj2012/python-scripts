#encoding: utf-8
'''
1. 重构darknet.py文件中单图片的检测函数
2. 根据编译环境获得的libdarknet.so文件选择该文件使用py2还是py3命令执行
3. 测试命令行 python2 darknet_ipl.py -i 0.jpg
'''

from ctypes import *
import math
import time
import random
import os, cv2
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]
class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]
class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]
class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]

class DARKNET():
    def __init__(self, so_path = './libdarknet.so', configPath = "./cfg/yolov4.cfg", weightPath = "yolov4.weights",\
                 metaPath= "./cfg/coco.data"):
        self.lib = CDLL(so_path, RTLD_GLOBAL)
        self.__loadFun__()
        set_gpu = self.lib.cuda_set_device  #GPU设置
        set_gpu.argtypes = [c_int]

        #加载模型文件和label文件
        self.netMain = self.__loadNet__(configPath, weightPath)
        self.metaMain = self.__loadMeta__(metaPath.encode("ascii"))
        self.altNames = self.__loadLabelNames__(metaPath)

        #检测阈值
        self.thresh = .1
        self.hier_thresh = .5
        self.nms = .45
    def __loadNet__(self, configPath, weightPath):
        load_net_custom = self.lib.load_network_custom
        load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        load_net_custom.restype = c_void_p
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        return netMain

    def __loadMeta__(self, metaPath):
        load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA
        metaMain = load_meta(metaPath)
        return metaMain

    def __loadLabelNames__(self, metaPath):
        altNames = list()
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
        return altNames

    def __loadFun__(self):
        #网络参数
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int
        #预测函数
        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)
        #
        self.predict_image_letterbox = self.lib.network_predict_image_letterbox
        self.predict_image_letterbox.argtypes = [c_void_p, IMAGE]
        self.predict_image_letterbox.restype = POINTER(c_float)
        #预测box
        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)
        #nms
        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
        #释放结果
        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

    def arrayToImage(self, arr):
        # need to return old values to avoid python freeing memory
        arr = arr.transpose(2, 0, 1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(w, h, c, data)
        return im, arr

    def detect_image(self, img, debug=False):
        custom_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #custom_image = cv2.resize(custom_image,(self.lib.network_width(self.netMain), self.lib.network_height(self.netMain)), interpolation = cv2.INTER_LINEAR)
        # import scipy.misc
        # custom_image = scipy.misc.imread(image)
        im, arr = self.arrayToImage(custom_image)		# you should comment line below: free_image(im)
        num = c_int(0) #检测数量
        pnum = pointer(num)
        self.predict_image(self.netMain, im)
        letter_box = 0
        # self.predict_image_letterbox(net, im)
        # letter_box = 1
        if debug: print("did prediction")
        # dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
        dets = self.get_network_boxes(self.netMain, im.w, im.h, self.thresh, self.hier_thresh, None, 0, pnum, letter_box)
        if debug: print("Got dets")
        num = pnum[0]
        if debug: print("got zeroth index of pnum")
        print(self.nms, type(self.nms))
        if self.nms:
            self.do_nms_sort(dets, num, self.metaMain.classes, self.nms)
        if debug: print("did sort")
        res = []
        if debug: print("about to range")
        for j in range(num):
            if debug: print("Ranging on " + str(j) + " of " + str(num))
            if debug: print("Classes: " + str(self.metaMain), self.metaMain.classes, self.metaMain.names)
            for i in range(self.metaMain.classes):
                if debug: print("Class-ranging on " + str(i) + " of " + str(self.metaMain.classes) + "= " + str(dets[j].prob[i]))
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.altNames is None:
                        nameTag = self.metaMain.names[i]
                    else:
                        nameTag = self.altNames[i]
                    if debug:
                        print("Got bbox", b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        if debug: print("did range")
        res = sorted(res, key=lambda x: -x[1])
        if debug: print("did sort")
        self.free_detections(dets, num)
        if debug: print("freed detections")
        return res

def parseArgs():
    args = argparse.ArgumentParser()
    args.add_argument("-i", '--images', help="image")
    return vars(args.parse_args())

def getAllFiles(folder):
    rtn = list()
    for dirname, _, imgs in os.walk(folder):
        for img in imgs:
            rtn.append(os.path.join(dirname, img))
    return rtn

if __name__ == "__main__":
    args = parseArgs()
    input = args['images']
    if os.path.isfile(input):
        img_paths = [input]
    else:
        img_paths = getAllFiles(input)
    detObj = DARKNET(so_path = './libdarknet.so', configPath = "./cfg/yolov4.cfg", weightPath = "yolov4.weights",\
                 metaPath= "./cfg/coco.data")
    roi_labels = ['motorbike', 'bicycle', 'person', 'car', 'bus', 'truck']

    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        tm_start = time.time()
        detections = detObj.detect_image(image) #内部利用cv函数将数据转换为rgb格式，并进行缩放
        tm_svc = time.time() - tm_start

        print('{} tm: {}, nums: {}'.format(img_name, tm_svc, str(len(detections))))
        for detection in detections:
            label = detection[0] #物体类别
            confidence = detection[1] #识别置信度
            bounds = detection[2] #中心点坐标和宽高信息
            pstring = label + ": " + str(np.rint(100 * confidence)) + "% : " + str(bounds)
            print(pstring)
            if label not in roi_labels:
                continue
            x1 = int(bounds[0] - bounds[2] / 2)
            y1 = int(bounds[1] - bounds[3] / 2)
            x2 = int(bounds[0] + bounds[2] / 2)
            y2 = int(bounds[1] + bounds[3] / 2)
            print(x1, y1, x2, y2)
            shape = image.shape
            boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
            cv2.rectangle(image, (x1,y1), (x2, y2), boxColor, 3)
            cv2.putText(image,label, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, boxColor)

        cv2.imwrite(os.path.join('rst', img_name), image)
