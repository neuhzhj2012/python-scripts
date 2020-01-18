#encoding: utf-8
'''
voc格式检测数据增强脚本
增强算法模块seq会出现崩溃情况，只能多开进程先用了
'''
import cv2, os
import gevent
import imageio
import imgaug as ia
import numpy as np
from lxml import etree, objectify
import xml.etree.ElementTree as ET
from multiprocessing import Process
from gevent import monkey
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
ia.seed(1)
monkey.patch_all()

def getXmls(root_folder):
    return [k for k in os.listdir(root_folder) if k.endswith('xml')==True]

def writeXmlRoot(anno):
    #写基础字段
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(    #根目录
        E.folder(anno['src']),        #根目录内容
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
        E.name(anno['class_name']),  #根目录内容
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

def parseXml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    basic_info = dict()
    folder = root.find('folder').text
    name = root.find('filename').text
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    img_depth = int(size.find('depth').text)
    basic_info['src']=folder
    basic_info['filename'] = name
    basic_info['width'] = img_w
    basic_info['height'] = img_h
    basic_info['channel'] = img_depth
    class_loc=dict()
    for obj in root.iter('object'):
        cls = obj.find('name').text
        # if cls not in classes :
        #     continue
        if cls not in class_loc.keys():
            class_loc[cls]=list()
        xmlbox = obj.find('bndbox')
        x1 = int(float(xmlbox.find('xmin').text))
        x2 = int(float(xmlbox.find('xmax').text))
        y1 = int(float(xmlbox.find('ymin').text))
        y2 = int(float(xmlbox.find('ymax').text))
        xmin=min(x1, x2)
        ymin=min(y1, y2)
        xmax=max(x1, x2)
        ymax=max(y1, y2)
        x1=max(0, xmin)
        y1=max(0, ymin)
        x2=min(img_w, xmax)
        y2=min(img_h, ymax)
        class_loc[cls].append([x1,y1,x2,y2])
    return basic_info,class_loc

def parseXmlOne(xml_name,xml_folder, img_folder,dst_folder, flag_saveRoiXml = False, flag_cropNotDraw=True):
    try:
        xml_abspath = os.path.join(xml_folder, xml_name)
        basic_info, parts = parseXml(xml_abspath)
        assert len(parts) != 0,"##{} has no objects##".format(xml_name)

        img_name = xml_name.replace('xml', 'jpg')
        img_abspath = os.path.join(img_folder, img_name)
        if not os.path.exists(img_abspath):
            img_abspath = img_abspath.replace('jpg', 'JPG')

        image = imageio.imread(img_abspath)
        img_h, img_w = image.shape[:2]
        boxes = list()
        for obj, locs in parts.items():
            for loc in locs:
                xmin, ymin, xmax, ymax = loc
                boxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=obj)) #保存原始标注数据

        bbs = BoundingBoxesOnImage(boxes, shape=image.shape)

        for i in range(num_aug):
            img_modify_name = str(i) + '_' + os.path.basename(img_abspath) #增强后的文件名称

            basic_info['filename'] = img_modify_name
            anno_tree = writeXmlRoot(basic_info)
            image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
            flag_valid_obj = False
            for box in bbs_aug.bounding_boxes:
                xmin, ymin, xmax, ymax = box.x1_int, box.y1_int, box.x2_int, box.y2_int
                flag_min_invald = ((abs(xmin-(img_w - 1)) < 10) or (abs(ymin-(img_h - 1)) < 10))
                flag_max_invald = ((xmax < 0) or (ymax < 0))
                flag_invald_value = ((xmin >= xmax) or (ymin >= ymax))
                flag_small = ((abs(xmax - xmin) < 5) or (abs(ymax - ymin) < 5))
                if flag_min_invald or flag_min_invald or flag_invald_value or flag_small: #翻转后bbs超出原图像大小
                    continue
                xmin = max(xmin, 0)
                ymin = max(ymin, 0)
                xmax = min(xmax, img_w - 1)
                ymax = min(ymax, img_h - 1)
                flag_valid_obj = True
                label = box.label
                object = dict()
                object['class_name'] = label
                object['bndbox'] = list()
                object['bndbox'] = [xmin, ymin, xmax, ymax]
                anno_tree.append(writeXmlSubRoot(object, bbox_type='xyxy'))
            dst_xml_abspath = os.path.join(dst_folder, 'xmls' ,str(i) + '_' + xml_name)
            dst_imgs_abspath = os.path.join(dst_folder, 'imgs' ,img_modify_name)
            if not os.path.exists(os.path.dirname(dst_xml_abspath)):
                os.makedirs(os.path.dirname(dst_xml_abspath))
            if not os.path.exists(os.path.dirname(dst_imgs_abspath)):
                os.makedirs(os.path.dirname(dst_imgs_abspath))
            if flag_valid_obj:
                ia.cv2.imencode('.jpg', image_aug[:, :, ::-1])[1].tofile(dst_imgs_abspath)
                writeXml(anno_tree, dst_xml_abspath)

    except:
        print ('##{} error##', xml_name)

def process_start(xml_list,xml_folder, img_folder,dst_folder, flag_saveRoiXml=True, flag_cropNotdraw=True):
    tasks = []
    for idx, xmlinfo in enumerate(xml_list):
        tasks.append(gevent.spawn(parseXmlOne, xmlinfo,xml_folder, img_folder,dst_folder, flag_saveRoiXml, flag_cropNotdraw))
    gevent.joinall(tasks)  # 使用协程来执行

def task_start(filepaths, batch_size=5, xml_folder='./Annotations', img_folder='JPEGImages', dst_folder='./tmp', flag_saveRoiXml=False, flag_cropNotdraw=True):  # 每batch_size条filepaths启动一个进程
    num=len(filepaths)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for idx in range(num // batch_size):
        url_list = filepaths[idx * batch_size:(idx + 1) * batch_size]
        p = Process(target=process_start, args=(url_list,xml_folder, img_folder,dst_folder,flag_saveRoiXml, flag_cropNotdraw,))
        p.start()

    if num % batch_size > 0:
        idx = num // batch_size
        url_list = filepaths[idx * batch_size:]
        p = Process(target=process_start, args=(url_list, xml_folder, img_folder, dst_folder,flag_saveRoiXml, flag_cropNotdraw,))
        p.start()




if __name__ == '__main__':
    xml_folder = 'Annotations' #xml文件夹
    img_folder = 'JPEGImages'   #图片文件夹
    dst_folder = 'cropImg'      #保存数据的根目录
    num_aug = 3 #单张图待增强的数据

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    xmls = getXmls(xml_folder)

    seq = iaa.Sequential([
        # iaa.DirectedEdgeDetect(alpha=(0.2, 0.4), direction=(0.0, 1.0)), #噪声
        iaa.GammaContrast((0.8, 1.2), per_channel=True),
        iaa.OneOf([
            # iaa.Multiply((1, 1.5), per_channel=0.5),
            # iaa.MultiplyHueAndSaturation((0.8, 1.2)),
            iaa.FrequencyNoiseAlpha(
                exponent=(-4, 0),
                first=iaa.Multiply((0.5, 1.5), per_channel=True),
                second=iaa.LinearContrast((0.5, 2.0))
            )
        ]),  # 颜色
        iaa.Fliplr(0.5),  # 水平翻转
        iaa.Flipud(0.01),  # 上下翻转
        iaa.CropAndPad(
            percent=(-0.05, 0.1),  # 依据图像宽高的[-5%, 10%]缩放，
            # pad_mode=ia.ALL, #若数值过小则填充时会映射出roi区域
            pad_mode="constant",
            pad_cval=(0, 255)
        ),
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # 原图缩放比例
            translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},  # 平移比例
            rotate=(-5, 5),  # 旋转角度，过大则会导致aug后的标注框过大
            # shear=(-16, 16), # 扭转角度，对刚体不可用
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)，插值方式
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            mode="constant"
        ),
    ])

    task_start(xmls, 5, xml_folder, img_folder, dst_folder)
    exit(-1)

    print('num: ', len(xmls))
    for xml_name in xmls:
        xml_abspath = os.path.join(xml_folder, xml_name)
        basic_info, parts = parseXml(xml_abspath)
        assert len(parts) != 0,"##{} has no objects##".format(xml_name)

        img_name = xml_name.replace('xml', 'jpg')
        img_abspath = os.path.join(img_folder, img_name)
        if not os.path.exists(img_abspath):
            img_abspath = img_abspath.replace('jpg', 'JPG')

        image = imageio.imread(img_abspath)
        img_h, img_w = image.shape[:2]

        boxes = list()
        for obj, locs in parts.items():
            for loc in locs:
                xmin, ymin, xmax, ymax = loc
                boxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=obj)) #保存原始标注数据

        bbs = BoundingBoxesOnImage(boxes, shape=image.shape)

        for i in range(num_aug):
            img_modify_name = str(i) + '_' + os.path.basename(img_abspath) #增强后的文件名称

            basic_info['filename'] = img_modify_name
            anno_tree = writeXmlRoot(basic_info)
            image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
            flag_valid_obj = False
            for box in bbs_aug.bounding_boxes:
                xmin, ymin, xmax, ymax = box.x1_int, box.y1_int, box.x2_int, box.y2_int
                flag_min_invald = ((abs(xmin-(img_w - 1)) < 10) or (abs(ymin-(img_h - 1)) < 10))
                flag_max_invald = ((xmax < 0) or (ymax < 0))
                flag_invald_value = ((xmin >= xmax) or (ymin >= ymax))
                flag_small = ((abs(xmax - xmin) < 5) or (abs(ymax - ymin) < 5))
                if flag_min_invald or flag_min_invald or flag_invald_value or flag_small: #翻转后bbs超出原图像大小
                    continue
                xmin = max(xmin, 0)
                ymin = max(ymin, 0)
                xmax = min(xmax, img_w - 1)
                ymax = min(ymax, img_h - 1)
                flag_valid_obj = True
                label = box.label
                object = dict()
                object['class_name'] = label
                object['bndbox'] = list()
                object['bndbox'] = [xmin, ymin, xmax, ymax]
                anno_tree.append(writeXmlSubRoot(object, bbox_type='xyxy'))
            dst_xml_abspath = os.path.join(dst_folder, 'xmls' ,str(i) + '_' + xml_name)
            dst_imgs_abspath = os.path.join(dst_folder, 'imgs' ,img_modify_name)
            if not os.path.exists(os.path.dirname(dst_xml_abspath)):
                os.makedirs(os.path.dirname(dst_xml_abspath))
            if not os.path.exists(os.path.dirname(dst_imgs_abspath)):
                os.makedirs(os.path.dirname(dst_imgs_abspath))
            if flag_valid_obj:
                ia.cv2.imencode('.jpg', image_aug[:, :, ::-1])[1].tofile(dst_imgs_abspath)
                writeXml(anno_tree, dst_xml_abspath)
