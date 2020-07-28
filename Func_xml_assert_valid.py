#encoding: utf-8
import os
import cv2
import gevent
import numpy as np
from lxml import etree, objectify
import xml.etree.ElementTree as ET
from multiprocessing import Process
from gevent import monkey

monkey.patch_all()
classes = ['car', 'bus', 'bicycle', 'motorbike', 'person']
classes = ['dadeng','houshijing','zhongwang','bashou','jinqikou',\
           'weideng','qianwudeng','fengdang','luntai','youxiangkou',\
           'kongzhitai','yibiaopan','hpfengkou','fangxiangpan',\
           'biansugan','hpkongtiao','hpyejingping']

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

def parseXml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    basic_info = dict()

    folder = root.find('folder').text
    name = root.find('filename').text
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    img_depth = 3
    basic_info['src']=folder
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
        if (x1==x2) or (y1==y2):
            continue
        class_loc[cls].append([x1,y1,x2,y2])

    return basic_info,class_loc

def parseXmlOne(xml_name,xml_folder, dst_folder):
    try:
        xml_abspath = os.path.join(xml_folder, xml_name)
        basic_info, parts = parseXml(xml_abspath)
        assert len(parts) != 0,"##{} has no objects##".format(xml_name)

        anno_tree = writeXmlRoot(basic_info) #基础信息

        for obj, locs in parts.items():  # 单个label
            for idx, loc in enumerate(locs):
                xmin, ymin, xmax, ymax = loc
                object = dict()
                object['name'] = obj
                object['bndbox'] = list()
                object['bndbox'] = [xmin, ymin, xmax, ymax]
                anno_tree.append(writeXmlSubRoot(object, bbox_type='xyxy'))

        writeXml(anno_tree, os.path.join(dst_folder, xml_name))

    except:
        print ('##{} error##', xml_name)

def process_start(xml_list,xml_folder, dst_folder):
    tasks = []
    for idx, xmlinfo in enumerate(xml_list):
        tasks.append(gevent.spawn(parseXmlOne, xmlinfo,xml_folder, dst_folder))
    gevent.joinall(tasks)  # 使用协程来执行

def task_start(filepaths, batch_size=5, xml_folder='./Annotations', dst_folder='./tmp'):  # 每batch_size条filepaths启动一个进程
    num=len(filepaths)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for idx in range(num // batch_size):
        url_list = filepaths[idx * batch_size:(idx + 1) * batch_size]
        p = Process(target=process_start, args=(url_list,xml_folder, dst_folder,))
        p.start()

    if num % batch_size > 0:
        idx = num // batch_size
        url_list = filepaths[idx * batch_size:]
        p = Process(target=process_start, args=(url_list, xml_folder, dst_folder,))
        p.start()

if __name__=='__main__':
    xml_folder = 'v0_error'
    dst_folder = 'v0_right'

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    xmls = getXmls(xml_folder)


    task_start(xmls, 1000, xml_folder, dst_folder)
    exit()
    #调试
    for xml_name in xmls:
        xml_abspath=os.path.join(xml_folder, xml_name)
        basic_info, parts = parseXml(xml_abspath)
        if len(parts)==0:
            print ("##{} has no objects##".format(xml_name))
            continue
        anno_tree = writeXmlRoot(basic_info) #基础信息


        for obj, locs in parts.items():  # 单个label
            for idx, loc in enumerate(locs):
                xmin, ymin, xmax, ymax = loc
                object = dict()
                object['name'] = obj
                object['bndbox'] = list()
                object['bndbox'] = [xmin, ymin, xmax, ymax]
                anno_tree.append(writeXmlSubRoot(object, bbox_type='xyxy'))

        writeXml(anno_tree, os.path.join(dst_folder, xml_name))
        break

