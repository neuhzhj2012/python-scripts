# encoding: utf-8
import requests
from multiprocessing import Process
import gevent
import sys
import os
import numpy
import urllib.request as urllib
import time
from gevent import monkey
import xml.etree.ElementTree as ET
from lxml import etree, objectify

monkey.patch_all()

ids = [1, 3, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 22, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 102, 111, 112, 114, 115, 119, 120, 121, 122, 124, 126, 129, 135, 136, 137, 139, 140, 141, 142, 143, 144, 145, 149, 150, 151, 152, 154, 155, 156, 157, 158, 159, 161, 162, 163, 164, 167, 168, 169, 171, 172, 173, 175, 181, 182, 183, 184, 186, 188, 190, 192, 193, 196, 197, 199, 200, 202, 203, 204, 205, 206, 210, 213, 214, 215, 217, 219, 220, 221, 222, 225, 226, 228, 231, 232, 233, 235, 238, 239, 240, 242, 244, 245, 248, 249, 250, 251, 252, 253, 255, 261, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 275, 277, 278, 279, 280, 281, 282, 283, 284, 286, 290, 293, 295, 296, 297, 298]

def parseXml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    basic_info = dict()

    name = root.find('filename').text
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    img_depth = int(size.find('depth').text)
    basic_info['filename'] = name
    basic_info['width'] = img_w
    basic_info['height'] = img_h
    basic_info['channel'] = img_depth

    class_loc = dict()
    for obj in root.iter('object'):
        cls = obj.find('name').text

        if int(cls) not in ids:
            continue
        if cls not in class_loc.keys():
            class_loc[cls] = list()
        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        x2 = int(xmlbox.find('xmax').text)
        y1 = int(xmlbox.find('ymin').text)
        y2 = int(xmlbox.find('ymax').text)
        xmin = min(x1, x2)
        ymin = min(y1, y2)
        xmax = max(x1, x2)
        ymax = max(y1, y2)
        x1 = max(0, xmin)
        y1 = max(0, ymin)
        x2 = min(img_w, xmax)
        y2 = min(img_h, ymax)
        if (x1 == x2) or (y1 == y2):
            continue
        class_loc[cls].append([x1, y1, x2, y2])

    return basic_info, class_loc

def fetch(url, subfolder, folder):
    name=os.path.basename(url)
    #if len(name.split('.')) ==1:
        #name +='.jpg'
    abspath=os.path.join(folder, name + '.txt')
    if not os.path.exists(os.path.dirname(abspath)):
        os.makedirs(os.path.dirname(abspath))
    
    try:
        tm_start=time.time()

        xml_path = os.path.join('Annotations', name+'.xml')
        basic_info, label_boxes = parseXml(xml_path)
        img_w = basic_info['width']
        img_h = basic_info['height']
        for label, boxes in label_boxes.items():
            for box in boxes:
                ct_x = (box[0] + box[2])/2
                ct_y = (box[1] + box[3])/2
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                buff = '{} {} {} {} {}\n'.format(ids.index(int(label)), ct_x*1.0/img_w,\
                                               ct_y * 1.0/img_h, box_w*1.0/img_w, box_h*1.0/img_h)
                print(buff)
                with open(abspath,'a+') as fp:
                    fp.write(buff)

        tm_end = time.time()
        print ('name: {} done, tm_svc: {}'.format(name, tm_end-tm_start))
    except Exception as e:
        print ("####name: {}, error: {}####".format(name,e))

        if os.path.exists(abspath):
            os.remove(abspath)

def process_start(url_list, folder):
    tasks = []
    for idx, urlinfo in enumerate(url_list):
        
        url=urlinfo.decode('utf-8').split(' ')[0]
        subfolder=''
        #subfolder=urlinfo.decode('utf-8').split(' ')[1]
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
    img_list = open('merge.csv', 'rb').readlines()
    imgs = [img.strip() for img in img_list]
    
    rstdir='carProd'
    tm_start = time.time()
    img_step = 1000
    task_start(imgs, img_step, rstdir)
    exit()
    #分阶段下载至不同文件夹
    for idx in range(0, len(imgs),img_step):
        urls = imgs[idx].split()[0]
        angle = imgs[idx].split()[1]
        urls_hdfs=[img.split()[0] for img in imgs[idx:idx+img_step]]

        rootdir = os.path.join(rstdir, angle)
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        
        task_start(urls_hdfs, 1000, rootdir)
        break
    tm_end = time.time()    
    print ('tm_task: {}'.format(tm_end - tm_start))


