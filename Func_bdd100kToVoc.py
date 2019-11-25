# encoding: utf-8
import requests
import json
from lxml import etree, objectify
import xml.etree.ElementTree as ET
from multiprocessing import Process
import gevent
import sys
import cv2
import os
import numpy
import urllib.request as urllib
import time
from gevent import monkey

monkey.patch_all()

class_rois=['car', 'bus', 'truck', 'person', 'rider', 'bike', 'motor']

basic_info = dict()
basic_info['width'] = 1280  # bdd100k固定分辨率
basic_info['height'] = 720
basic_info['channel'] = 3

def writeXmlRoot(anno):
    #写基础字段
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(    #根目录
        # E.folder(anno['src']),        #根目录内容
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

def fetch(info, folder):
    try:
        name = info['name']
        attrs_time = info['attributes']['timeofday']  # daytime, night
        flag_xml = False
        rois = dict()
        for roi_info in info['labels']:
            label = roi_info['category']
            if label not in class_rois:
                continue
            flag_xml = True
            # if 'box2d' in roi_info.keys():
            xxyy = list(roi_info['box2d'].values())
            if label not in rois.keys():
                rois[label]=list()
            rois[label].append(xxyy)
        if flag_xml:
            dst_folder = os.path.join(folder, attrs_time)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            basic_info['filename']=name
            for obj, locs in rois.items():  #扩充roi区域并生成新xml文件
                for idx, loc in enumerate(locs):
                    xmin, ymin, xmax, ymax = loc
                    object = dict()
                    object['class_name'] = obj
                    object['bndbox'] = list()
                    object['bndbox'] = [xmin, ymin, xmax, ymax]
                    anno_tree = writeXmlRoot(basic_info)
                    anno_tree.append(writeXmlSubRoot(object, bbox_type='xyxy'))
            writeXml(anno_tree, os.path.join(dst_folder, name.replace('jpg', 'xml')))
    except Exception as e:
        print ("####name: {}, error: {}####".format(info,e))

def process_start(url_list, folder):
    tasks = []
    for idx, urlinfo in enumerate(url_list):
        # url=urlinfo.decode('utf-8')
        abspath=urlinfo
        tasks.append(gevent.spawn(fetch, abspath, folder))
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
    train_labels = 'bdd100k_labels_images_train.json'

    with open(train_labels) as fp:
        data = json.load(fp)

    rstdir='carProd' #xml
    img_step = 1000
    task_start(data, img_step, rstdir)
    exit()
    #分阶段下载至不同文件夹
    for info in data:
        name = info['name']
        attrs_time = info['attributes']['timeofday']  # daytime, night
        flag_xml = False
        rois = dict()
        for roi_info in info['labels']:
            label = roi_info['category']
            if label not in class_rois:
                continue
            flag_xml = True
            # if 'box2d' in roi_info.keys():
            xxyy = list(roi_info['box2d'].values())
            if label not in rois.keys():
                rois[label] = list()
            rois[label].append(xxyy)
        if flag_xml:
            dst_folder = os.path.join(rstdir, attrs_time)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            basic_info['filename'] = name
            for obj, locs in rois.items():  # 扩充roi区域并生成新xml文件
                for idx, loc in enumerate(locs):
                    xmin, ymin, xmax, ymax = loc
                    object = dict()
                    object['class_name'] = obj
                    object['bndbox'] = list()
                    object['bndbox'] = [xmin, ymin, xmax, ymax]
                    anno_tree = writeXmlRoot(basic_info)
                    anno_tree.append(writeXmlSubRoot(object, bbox_type='xyxy'))
            writeXml(anno_tree, os.path.join(dst_folder, name.replace('jpg', 'xml')))
        break



