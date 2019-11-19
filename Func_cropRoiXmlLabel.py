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
classes = ['dadeng']

def colormap(rgb=False):
    color_list = np.array(
        [
            255, 0, 0,
            255, 255, 0,
            0, 0, 255,
            255, 0, 255,
            218, 112, 214,
            50, 205, 50,
            255, 192, 203,
            0, 139, 139,
            219, 112, 147,
            218, 165, 32,
            0, 255, 255,
            255, 20, 147,
            255, 165, 0,
            0, 0, 139,
            128, 0, 128,
            95, 158, 160,
            148, 0, 211,
            100, 149, 237,
            123, 104, 238,
            135, 206, 235,
            127, 255, 170,
            255, 99, 71,
            205, 133, 63,
            205, 92, 92,
            255, 215, 0,
            220, 20, 60
        ]
    ).astype(np.uint8)
    color_list = color_list.reshape((-1, 3))
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

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
        class_loc[cls].append([x1,y1,x2,y2])

    return basic_info,class_loc

def parseXmlOne(xml_name,xml_folder, img_folder,dst_folder, flag_saveRoiXml = False, flag_cropNotDraw=True):
    try:
        xml_abspath = os.path.join(xml_folder, xml_name)
        basic_info, parts = parseXml(xml_abspath)
        assert len(parts) != 0,"##{} has no objects##".format(xml_name)

        w_buff = 10
        h_buff = 5

        if flag_saveRoiXml:
            # anno_tree = writeXmlRoot(basic_info)
            img_name = xml_name.replace('xml', 'jpg')
            img_abspath = os.path.join(img_folder, img_name)
            if not os.path.exists(img_abspath):
                img_abspath = img_abspath.replace('jpg', 'JPG')
            img = cv2.imread(img_abspath)
            for obj, locs in parts.items():
                for idx, loc in enumerate(locs):
                    dst_name = xml_name[:-4] + "_" + str(idx) + '.jpg'
                    basic_info['filename'] = dst_name

                    dst_abspath = os.path.join(dst_folder, obj, dst_name)
                    xmin, ymin, xmax, ymax = loc
                    w_box = xmax - xmin
                    h_box = ymax - ymin
                    xmin_crop = int(max(0, xmin - w_box / 2))
                    ymin_crop = int(max(0, ymin - h_box / 2))
                    xmax_crop = int(min(basic_info['width'], xmax + w_box / 2))
                    ymax_crop = int(min(basic_info['height'], ymax + h_box / 2))

                    img_crop = img[ymin_crop:ymax_crop, xmin_crop:xmax_crop, :]
                    cv2.imwrite(dst_abspath, img_crop)

                    xmax -= xmin_crop
                    ymax -= ymin_crop
                    xmin -= xmin_crop
                    ymin -= ymin_crop
                    basic_info['width'] = xmax_crop - xmin_crop + 1
                    basic_info['height'] = ymax_crop - ymin_crop + 1

                    object = dict()
                    object['class_name'] = obj
                    object['bndbox'] = list()
                    object['bndbox'] = [xmin, ymin, xmax, ymax]
                    anno_tree = writeXmlRoot(basic_info)
                    anno_tree.append(writeXmlSubRoot(object, bbox_type='xyxy'))
                    writeXml(anno_tree, os.path.join(dst_folder, dst_name.replace('jpg', 'xml')))

            # writeXml(anno_tree, os.path.join(dst_folder, xml_name))
            if flag_cropNotDraw: # 保存单个目标区域
                # for obj, locs in parts.iteritems(): #py2
                for obj, locs in parts.items():
                    for idx, loc in enumerate(locs):
                        dst_name = xml_name[:-4] + "_" + str(idx) + '.jpg'
                        dst_abspath = os.path.join(dst_folder, obj, dst_name)
                        x1, y1, x2, y2 = loc
                        img_crop = img[y1:y2, x1:x2, :]
                        cv2.imwrite(dst_abspath, img_crop)
            else:
                # for obj, locs in parts.iteritems(): #py2
                for obj, locs in parts.items():
                    dst_name = xml_name[:-4] + '.jpg'
                    dst_abspath = os.path.join(dst_folder, dst_name)
                    color = cmap[classes.index(obj)%len(classes)].astype( np.uint8).tolist()
                    for idx, loc in enumerate(locs):
                        x1, y1, x2, y2 = loc
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.imwrite(dst_abspath, img)
        else:
            img_name = xml_name.replace('xml', 'jpg')
            img_abspath = os.path.join(img_folder, img_name)
            if not os.path.exists(img_abspath):
                img_abspath = img_abspath.replace('jpg', 'JPG')
            img = cv2.imread(img_abspath)

            if flag_cropNotDraw: # 保存单个目标区域
                # for obj, locs in parts.iteritems(): #py2
                for obj, locs in parts.items():
                    for idx, loc in enumerate(locs):
                        dst_name = xml_name[:-4] + "_" + str(idx) + '.jpg'
                        dst_abspath = os.path.join(dst_folder, obj, dst_name)
                        x1, y1, x2, y2 = loc
                        img_crop = img[y1:y2, x1:x2, :]
                        cv2.imwrite(dst_abspath, img_crop)
            else:
                # for obj, locs in parts.iteritems(): #py2
                for obj, locs in parts.items():
                    dst_name = xml_name[:-4] + '.jpg'
                    dst_abspath = os.path.join(dst_folder, dst_name)
                    color = cmap[classes.index(obj)%len(classes)].astype( np.uint8).tolist()
                    for idx, loc in enumerate(locs):
                        x1, y1, x2, y2 = loc
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.imwrite(dst_abspath, img)

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

if __name__=='__main__':
    xml_folder = 'crop_xml'
    img_folder = 'crop_img'
    dst_folder = 'cropImg'

    flag_cropNotDraw = True  # 画整体结果，非裁剪成单独类别
    flag_saveRoiXml = True #是否保留感兴趣类别的xml文件
    #初始化存图路径
    if flag_cropNotDraw:
        for obj in classes:
            dst_abspath=os.path.join(dst_folder, obj)
            if not os.path.exists(dst_abspath):
                os.makedirs(dst_abspath)
    else:
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

    xmls = getXmls(xml_folder)
    #xmls = ['00043251505971_3832319_16.jpg.xml']
    cmap = colormap()

    # task_start(xmls, 1000, xml_folder, img_folder, dst_folder, flag_saveRoiXml, flag_cropNotDraw)
    # exit()
    #调试
    for xml_name in xmls:
        xml_abspath=os.path.join(xml_folder, xml_name)
        basic_info, parts = parseXml(xml_abspath)
        print (parts)
        if len(parts)==0:
            print ("##{} has no objects##".format(xml_name))
            continue

        if flag_saveRoiXml:
            # anno_tree = writeXmlRoot(basic_info)
            img_name = xml_name.replace('xml', 'jpg')
            img_abspath = os.path.join(img_folder, img_name)
            if not os.path.exists(img_abspath):
                img_abspath = img_abspath.replace('jpg', 'JPG')
            img = cv2.imread(img_abspath)
            for obj, locs in parts.items():
                for idx, loc in enumerate(locs):
                    dst_name = xml_name[:-4] + "_" + str(idx) + '.jpg'
                    basic_info['filename'] = dst_name

                    dst_abspath = os.path.join(dst_folder, obj, dst_name)
                    xmin, ymin, xmax, ymax = loc
                    w_box = xmax - xmin
                    h_box = ymax - ymin
                    xmin_crop = int(max(0, xmin - w_box / 2))
                    ymin_crop = int(max(0, ymin - h_box / 2))
                    xmax_crop = int(min(basic_info['width'], xmax + w_box / 2))
                    ymax_crop = int(min(basic_info['height'], ymax + h_box / 2))

                    img_crop = img[ymin_crop:ymax_crop, xmin_crop:xmax_crop, :]
                    cv2.imwrite(dst_abspath, img_crop)

                    xmax -= xmin_crop
                    ymax -= ymin_crop
                    xmin -= xmin_crop
                    ymin -= ymin_crop
                    basic_info['width'] = xmax_crop - xmin_crop + 1
                    basic_info['height'] = ymax_crop - ymin_crop + 1

                    object = dict()
                    object['class_name'] = obj
                    object['bndbox'] = list()
                    object['bndbox'] = [xmin, ymin, xmax, ymax]
                    anno_tree = writeXmlRoot(basic_info)
                    anno_tree.append(writeXmlSubRoot(object, bbox_type='xyxy'))
                    writeXml(anno_tree, os.path.join(dst_folder, dst_name.replace('jpg', 'xml')))

            # writeXml(anno_tree, os.path.join(dst_folder, xml_name))
            if flag_cropNotDraw: # 保存单个目标区域
                # for obj, locs in parts.iteritems(): #py2
                for obj, locs in parts.items():
                    for idx, loc in enumerate(locs):
                        dst_name = xml_name[:-4] + "_" + str(idx) + '.jpg'
                        dst_abspath = os.path.join(dst_folder, obj, dst_name)
                        x1, y1, x2, y2 = loc
                        img_crop = img[y1:y2, x1:x2, :]
                        cv2.imwrite(dst_abspath, img_crop)
            else:
                # for obj, locs in parts.iteritems(): #py2
                for obj, locs in parts.items():
                    dst_name = xml_name[:-4] + '.jpg'
                    dst_abspath = os.path.join(dst_folder, dst_name)
                    color = cmap[classes.index(obj)%len(classes)].astype( np.uint8).tolist()
                    for idx, loc in enumerate(locs):
                        x1, y1, x2, y2 = loc
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.imwrite(dst_abspath, img)
        else:
            img_name = xml_name.replace('xml', 'jpg')
            img_abspath = os.path.join(img_folder, img_name)
            if not os.path.exists(img_abspath):
                img_abspath = img_abspath.replace('jpg', 'JPG')
            img = cv2.imread(img_abspath)

            if flag_cropNotDraw: # 保存单个目标区域
                # for obj, locs in parts.iteritems(): #py2
                for obj, locs in parts.items():
                    for idx, loc in enumerate(locs):
                        dst_name = xml_name[:-4] + "_" + str(idx) + '.jpg'
                        dst_abspath = os.path.join(dst_folder, obj, dst_name)
                        x1, y1, x2, y2 = loc
                        img_crop = img[y1:y2, x1:x2, :]
                        cv2.imwrite(dst_abspath, img_crop)
            else:
                # for obj, locs in parts.iteritems(): #py2
                for obj, locs in parts.items():
                    dst_name = xml_name[:-4] + '.jpg'
                    dst_abspath = os.path.join(dst_folder, dst_name)
                    color = cmap[classes.index(obj)%len(classes)].astype( np.uint8).tolist()
                    for idx, loc in enumerate(locs):
                        x1, y1, x2, y2 = loc
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.imwrite(dst_abspath, img)