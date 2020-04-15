# encoding: utf-8
#py3，对指定classes类别的物体进行增强，包括augument和crop两种方法
import os, sys, cv2
import uuid
import time
import requests
import gevent
import imageio
import numpy as np
import xml.etree.ElementTree as ET
from gevent import monkey
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from multiprocessing import Process
from lxml import etree, objectify

ia.seed(1)
seq = iaa.Sequential([
         #iaa.DirectedEdgeDetect(alpha=(0.2, 0.4), direction=(0.0, 1.0)), #噪声
         iaa.GammaContrast((0.8, 1.2), per_channel=True),
         iaa.OneOf([
                    #iaa.Multiply((1, 1.5), per_channel=0.5),
                    #iaa.MultiplyHueAndSaturation((0.8, 1.2)),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),    #颜色
        iaa.Fliplr(0.5), # 水平翻转
        iaa.Flipud(0.01),  #上下翻转
        iaa.CropAndPad(
            percent=(-0.05, 0.1), #依据图像宽高的[-5%, 10%]缩放，
            #pad_mode=ia.ALL, #若数值过小则填充时会映射出roi区域
            pad_mode = "constant",
            pad_cval=(0, 255)
        ),

        iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # 原图缩放比例
                translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)}, # 平移比例
                rotate=(-5, 5), # 旋转角度，过大则会导致aug后的标注框过大
                #shear=(-16, 16), # 扭转角度，对刚体不可用
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)，插值方式
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                mode="constant"
           ),
])
monkey.patch_all()

classes=['154', '193', '253', '136', '255']
def parseXml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    basic_info = dict()

    name = root.find('filename').text
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    img_depth = int(size.find('depth').text)
    basic_info['folder']='onlineV1'
    basic_info['filename'] = name
    basic_info['width'] = img_w
    basic_info['height'] = img_h
    basic_info['channel'] = img_depth

    class_loc=dict()
    for obj in root.iter('object'):
        cls = obj.find('name').text

        if cls not in classes :
            continue
        if cls not in class_loc.keys():
            class_loc[cls]=list()
        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        x2 = int(xmlbox.find('xmax').text)
        y1 = int(xmlbox.find('ymin').text)
        y2 = int(xmlbox.find('ymax').text)
        xmin=min(x1, x2)
        ymin=min(y1, y2)
        xmax=max(x1, x2)
        ymax=max(y1, y2)
        x1=max(0, xmin)
        y1=max(0, ymin)
        x2=min(img_w, xmax)
        y2=min(img_h, ymax)
        if (x1==x2) or (y1==y2):
            continue
        class_loc[cls].append([x1,y1,x2,y2])

    return basic_info,class_loc

def writeXmlRoot(anno):
    #写基础字段
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(    #根目录
        E.folder(anno['folder']),        #根目录内容
        E.filename(anno['filename']),
        E.size(
            E.width(anno['width']),  #子目录内容
            E.height(anno['height']),
            E.depth(anno['channel'])
        ),
    )
    return anno_tree  #anno_tree.append(writeXmlSubRoot(anno))

def writeXmlSubRoot(anno, bbox_type='xyxy'):
    #增加xml的子字段
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xywh':
        xmin, ymin, w, h = anno['bndbox']
        xmax = xmin+w
        ymax = ymin+h
    else:
        xmin, ymin, xmax, ymax = anno['bndbox']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.object(             #根目录
        E.name(anno['name']),  #根目录内容
        E.bndbox(
            E.xmin(xmin),             #子目录内容
            E.ymin(ymin),
            E.xmax(xmax),
            E.ymax(ymax)
        ),
    )
    return anno_tree

def writeXml(anno_tree, xml_name):
    etree.ElementTree(anno_tree).write(xml_name, pretty_print=True)

def augument(img_path, class_loc):
    image = imageio.imread(img_path)

    boxes_info = list()
    for obj, locs in class_loc.items():
        for idx, loc in enumerate(locs):
            xmin, ymin, xmax, ymax = loc

            boxes_info.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=obj))

    bbs = BoundingBoxesOnImage(boxes_info, shape=image.shape)

    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    class_loc_aug = dict()
    for box in bbs_aug.bounding_boxes:
        if box.label not in class_loc_aug.keys():
            class_loc_aug[box.label]=list()
        class_loc_aug[box.label].append([box.x1_int, box.y1_int, box.x2_int, box.y2_int])
    return image_aug, class_loc_aug

#根据物体位置，保留相对位置不变，随机裁剪整个图的边界
def crop(img_path, class_loc):
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    xmin_all = 10000
    ymin_all = 10000
    xmax_all = 0
    ymax_all = 0
    for obj, locs in class_loc.items():
        for idx, loc in enumerate(locs):
            xmin, ymin, xmax, ymax = loc
            xmin_all = min(xmin, xmin_all)
            ymin_all = min(ymin, ymin_all)
            xmax_all = max(xmax, xmax_all)
            ymax_all = max(ymax, ymax_all)

    xmin_ = np.random.choice(range(0, xmin_all), 1)[0]
    ymin_ = np.random.choice(range(0, ymin_all), 1)[0]
    xmax_ = np.random.choice(range(xmax_all, img_w), 1)[0]
    ymax_ = np.random.choice(range(ymax_all, img_h), 1)[0]

    img_crop = img[ymin_:ymax_, xmin_:xmax_,:]
    print(ymin_, ymax_, xmin_, xmax_, img_crop.shape)
    class_loc_crop = dict()

    for obj, locs in class_loc.items():
        if obj not in class_loc_crop.keys():
            class_loc_crop[obj] = list()
        for idx, loc in enumerate(locs):
            xmin, ymin, xmax, ymax = loc
            xmin -= xmin_
            ymin -= ymin_
            xmax -= xmin_
            ymax -= ymin_
            class_loc_crop[obj].append([xmin, ymin, xmax, ymax])
    return img_crop, class_loc_crop
def fetch(url, subfolder, folder):
    name=os.path.basename(url)
    abspath=os.path.join(folder, name)
    if not os.path.exists(os.path.dirname(abspath)):
        os.makedirs(os.path.dirname(abspath))
    
    try:
        basic_info,class_loc = parseXml(url)
        print (class_loc)
        if len(class_loc.keys())==0:
            return None

        img_name = basic_info['filename']
        img_path = os.path.join('/workspace/JPEGImages', img_name)
        print(img_name, img_path)

        if np.random.randint(10, 19)%3==0:
            img_op, class_loc_op = augument(img_path, class_loc)
            filename = 'aug_'+str(uuid.uuid4())[:5] + '_'+img_name
            ia.cv2.imwrite(os.path.join(folder, filename), img_op)

        else:
            img_op, class_loc_op = crop(img_path, class_loc)
            filename = 'crop_' + str(uuid.uuid4())[:5] + '_' + img_name
            cv2.imwrite(os.path.join(folder, filename), img_op)

        img_h, img_w = img_op.shape
        img_depth = 3
        basic_info['folder'] = 'onlineV1_aug'
        basic_info['filename'] = filename
        basic_info['width'] = img_w
        basic_info['height'] = img_h
        basic_info['channel'] = img_depth

        anno_tree = writeXmlRoot(basic_info) #基础信息

        for obj, locs in class_loc_op.items():  # 单个label
            for idx, loc in enumerate(locs):
                xmin, ymin, xmax, ymax = loc
                object = dict()
                object['name'] = obj
                object['bndbox'] = list()
                object['bndbox'] = [xmin, ymin, xmax, ymax]
                anno_tree.append(writeXmlSubRoot(object, bbox_type='xyxy'))

        if len(class_loc) > 0:
            xmlname = filename.replace('.jpg', '.xml')
            writeXml(anno_tree, os.path.join(folder, xmlname))
    except Exception as e:
        print ("####name: {}, error: {}####".format(name,e))

        if os.path.exists(abspath):
            os.remove(abspath)

def process_start(url_list, folder):
    tasks = []
    for idx, urlinfo in enumerate(url_list):
        
        url = urlinfo.decode('utf-8')
        subfolder='hehe'
        tasks.append(gevent.spawn(fetch, url,subfolder, folder))
    gevent.joinall(tasks)  # 使用协程来执行


def task_start(filepaths, batch_size=5, folder='./tmp'):  # 每batch_size条filepaths启动一个进程
    num=len(filepaths)

    if not os.path.exists(folder):
        os.makedirs(folder)

    for idx in range(num // batch_size):
        url_list = filepaths[idx * batch_size:(idx + 1) * batch_size]
        p = Process(target=process_start, args=(url_list, folder,))
        p.start()

    if num % batch_size > 0:
        idx = num // batch_size
        url_list = filepaths[idx * batch_size:]
        p = Process(target=process_start, args=(url_list, folder,))
        p.start()

if __name__ == '__main__':
    img_list = open('onlineV1_files.txt', 'r').readlines()
    imgs = [img.strip() for img in img_list]
    
    rstdir='onlineV1_VOC'
    tm_start = time.time()
    img_step = 1000
    #task_start(imgs, img_step, rstdir)
    #exit()
    #分阶段下载至不同文件夹
    folder = rstdir
    for url in imgs:
        name = os.path.basename(url)
        abspath = os.path.join(folder, name)
        if not os.path.exists(os.path.dirname(abspath)):
            os.makedirs(os.path.dirname(abspath))

        basic_info, class_loc = parseXml(url)
        print (class_loc)
        if len(class_loc.keys())==0:
            continue


        img_name = basic_info['filename']
        img_path = os.path.join('/workspace/JPEGImages', img_name)

        #if np.random.randint(10, 19) % 3 == 0:
        if False:
            img_op, class_loc_op = augument(img_path, class_loc)
            filename = 'aug_' + str(uuid.uuid4())[:5] + '_' + img_name
            ia.cv2.imwrite(os.path.join(folder, filename), img_op)

        else:
            img_op, class_loc_op = crop(img_path, class_loc)
            filename = 'crop_' + str(uuid.uuid4())[:5] + '_' + img_name
            cv2.imwrite(os.path.join(folder, filename), img_op)

        img_h, img_w = img_op.shape[:2]
        img_depth = 3
        basic_info['folder'] = 'onlineV1_aug'
        basic_info['filename'] = filename
        basic_info['width'] = img_w
        basic_info['height'] = img_h
        basic_info['channel'] = img_depth

        anno_tree = writeXmlRoot(basic_info)  # 基础信息

        for obj, locs in class_loc_op.items():  # 单个label
            for idx, loc in enumerate(locs):
                xmin, ymin, xmax, ymax = loc
                object = dict()
                object['name'] = obj
                object['bndbox'] = list()
                object['bndbox'] = [xmin, ymin, xmax, ymax]
                anno_tree.append(writeXmlSubRoot(object, bbox_type='xyxy'))

        if len(class_loc) > 0:
            xmlname = filename.replace('.jpg', '.xml')
            writeXml(anno_tree, os.path.join(folder, xmlname))


