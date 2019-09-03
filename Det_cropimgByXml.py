#encoding: utf-8
import os
import cv2
import gevent
import numpy as np
import xml.etree.ElementTree as ET
from multiprocessing import Process
from gevent import monkey

monkey.patch_all()

classes = ['dadeng','weideng','houshijing','qianwudeng','luntai','zhongwang',
           'fengdang','bashou','jinqikou','youxiangkou','kongzhitai','fangxiangpan',
           'yibiaopan','biansugan','hpkongtiao','hpfengkou', 'hpyejingping']

def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
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
        img = cv2.imread(img_abspath)

        assert img.shape[:2] == img_size, '##{} imgsize not match##'.format(xml_name)

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
                dst_name = xml_name[:-4] + '.jpg'
                dst_abspath = os.path.join(dst_folder, dst_name)
                color = cmap[classes.index(obj)%len(classes)].astype( np.uint8).tolist()
                for idx, loc in enumerate(locs):
                    x1, y1, x2, y2 = loc
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.imwrite(dst_abspath, img)

    except:
        print '##{} error##', xml_name

def process_start(xml_list,xml_folder, img_folder,dst_folder, flag_cropNotdraw=True):
    tasks = []
    for idx, xmlinfo in enumerate(xml_list):
        tasks.append(gevent.spawn(parseXmlOne, xmlinfo,xml_folder, img_folder,dst_folder, flag_cropNotdraw))
    gevent.joinall(tasks)  # 使用协程来执行

def task_start(filepaths, batch_size=5, xml_folder='./Annotations', img_folder='JPEGImages', dst_folder='./tmp', flag_cropNotdraw=True):  # 每batch_size条filepaths启动一个进程
    num=len(filepaths)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for idx in range(num / batch_size):
        url_list = filepaths[idx * batch_size:(idx + 1) * batch_size]
        p = Process(target=process_start, args=(url_list,xml_folder, img_folder,dst_folder,flag_cropNotdraw,))
        p.start()

    if num % batch_size > 0:
        idx = num / batch_size
        url_list = filepaths[idx * batch_size:]
        p = Process(target=process_start, args=(url_list, xml_folder, img_folder, dst_folder,flag_cropNotdraw,))
        p.start()

cmap = colormap()
if __name__=='__main__':
    xml_folder = 'Annotations'
    img_folder = 'JPEGImages'
    dst_folder = 'cropImg'

    flag_cropNotDraw = False  # 画整体结果，非裁剪成单独类别
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
            print '##{} error##', name

