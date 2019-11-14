#encoding: utf-8
import os
import cv2
import gevent
import numpy as np
import xml.etree.ElementTree as ET
from multiprocessing import Process
from gevent import monkey

monkey.patch_all()

classes = ['car']

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

def parseXml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

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
        x2=min(w, xmax)
        y2=min(h, ymax)
        class_loc[cls].append([x1,y1,x2,y2])
        '''
        for xmlbox in  obj.iter('bndbox'):
        #xmlbox = obj.find('bndbox')
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
            x2=min(w, xmax)
            y2=min(h, ymax)
            class_loc[cls].append([x1,y1,x2,y2])
        '''
    return (h,w),class_loc

def getXmls(root_folder):
    return [k for k in os.listdir(root_folder) if k.endswith('xml')==True]

def parseXmlOne(xml_name,xml_folder, img_folder,dst_folder, flag_cropNotDraw=True):
    try:
        xml_abspath = os.path.join(xml_folder, xml_name)
        img_size, parts = parseXml(xml_abspath)
        assert len(parts) != 0,"##{} has no objects##".format(xml_name)

        img_name = xml_name.replace('xml', 'jpg')
        img_abspath = os.path.join(img_folder, img_name)
        if not os.path.exists(img_abspath):
            img_abspath = img_abspath.replace('jpg', 'JPG')
        #img_abspath = os.path.join(img_folder, img_name)
        img = cv2.imread(img_abspath)

        assert img.shape[:2] == img_size, '##{} imgsize not match##'.format(xml_name)

        if flag_cropNotDraw:
            # 保存单个目标区域
            #for obj, locs in parts.iteritems():
            for obj, locs in parts.items():
                for idx, loc in enumerate(locs):
                    dst_name = xml_name[:-4] + "_" + str(idx) + '.jpg'
                    dst_abspath = os.path.join(dst_folder, obj, dst_name)
                    x1, y1, x2, y2 = loc
                    img_crop = img[y1:y2, x1:x2, :]
                    cv2.imwrite(dst_abspath, img_crop)
        else:
            for obj, locs in parts.items():
                dst_name = xml_name[:-4] + '.jpg'
                dst_abspath = os.path.join(dst_folder, dst_name)
                color = cmap[classes.index(obj)%len(classes)].astype( np.uint8).tolist()
                for idx, loc in enumerate(locs):
                    x1, y1, x2, y2 = loc
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.imwrite(dst_abspath, img)

    except Exception as e:
        print ('##{} error## {}'.format(xml_name, str(e)))

def process_start(xml_list,xml_folder, img_folder,dst_folder, flag_cropNotdraw=True):
    tasks = []
    for idx, xmlinfo in enumerate(xml_list):
        tasks.append(gevent.spawn(parseXmlOne, xmlinfo,xml_folder, img_folder,dst_folder, flag_cropNotdraw))
    gevent.joinall(tasks)  # 使用协程来执行

def task_start(filepaths, batch_size=5, xml_folder='./Annotations', img_folder='JPEGImages', dst_folder='./tmp', flag_cropNotdraw=True):  # 每batch_size条filepaths启动一个进程
    num=len(filepaths)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for idx in range(num // batch_size):
        url_list = filepaths[idx * batch_size:(idx + 1) * batch_size]
        p = Process(target=process_start, args=(url_list,xml_folder, img_folder,dst_folder,flag_cropNotdraw,))
        p.start()

    if num % batch_size > 0:
        idx = num // batch_size
        url_list = filepaths[idx * batch_size:]
        p = Process(target=process_start, args=(url_list, xml_folder, img_folder, dst_folder,flag_cropNotdraw,))
        p.start()

cmap = colormap()
if __name__=='__main__':
    xml_folder = 'Annotations'
    img_folder = 'JPEGImages'
    dst_folder = 'cropImg'

    flag_cropNotDraw = True  # 画整体结果，非裁剪成单独类别
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

    task_start(xmls, 1000, xml_folder, img_folder, dst_folder, flag_cropNotDraw)
    exit()

    for xml_name in xmls:
        try:
            xml_abspath=os.path.join(xml_folder, xml_name)
            img_size, parts = parseXml(xml_abspath)
            print (parts)
            if len(parts)==0:
                print ("##{} has no objects##".format(xml_name))
                continue

            img_name=name.replace('xml','jpg')
            img_abspath =os.path.join(img_folder, img_name)
            img = cv2.imread(img_abspath)

            assert img.shape[:2]==img_size,'##{} imgsize not match##'.format(name)

            if flag_cropNotDraw:
                # 保存单个目标区域
                for obj, locs in parts.iteritems():
                    for idx, loc in enumerate(locs):
                        dst_name = xml_name[:-4] + "_" + str(idx) + '.jpg'
                        dst_abspath = os.path.join(dst_folder, obj, dst_name)
                        x1, y1, x2, y2 = loc
                        img_crop = img[y1:y2, x1:x2, :]
                        cv2.imwrite(dst_abspath, img_crop)
            else:
                for obj, locs in parts.iteritems():
                    print ('{} num: {}'.format(obj, len(locs)))
                    dst_name = xml_name[:-4] + '.jpg'
                    dst_abspath = os.path.join(dst_folder, dst_name)
                    color = cmap[classes.index(obj) % len(classes)].astype(np.uint8).tolist()
                    for idx, loc in enumerate(locs):
                        x1, y1, x2, y2 = loc
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.imwrite(dst_abspath, img)
        except:
            print ('##{} error##', name)

