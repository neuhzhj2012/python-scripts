#encoding: utf-8
import cv2
import numpy as np
class CustomError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) #初始化父类
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo


def bytes2cv(img_bytes):
    img_np = np.asarray(bytearray(img_bytes), np.uint8)
    # img_tocv = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    img_tocv = cv2.imdecode(img_np, 1)
    return img_tocv

def isFastdfsAccess(fastdfs_id, whole_name, part_name):
    flag_post = False

    if fastdfs_id in whole_name:
        flag_post = True
    else:
        for name in part_name:
            if (fastdfs_id in name) and fastdfs_id == name[:len(fastdfs_id)]:
                flag_post = True
                break
    return flag_post

def isTop5Access(type_id, whole_name, part_name=[]):
    flag_post = False

    if type_id in whole_name:
        flag_post = True
    else:
        for name in part_name:
            if (type_id in name) and type_id == name[:len(type_id)]:
                flag_post = True
                break
    return flag_post


def getDfsUrl(img_url, url_part_prefix='autoimg.cn'): #避免http和https,及img[1,2,3]的影响
    if url_part_prefix in img_url:
        len_prefix = len(url_part_prefix)
        if 'dwdfs/' in img_url:
            len_prefix += len('dwdfs/')
        img_url = img_url[img_url.index(url_part_prefix) + len_prefix + 1:]
    return img_url

def getDfsWholeUrl(img_url, url_part_prefix='autoimg.cn'): #避免http和https,及img[1,2,3]的影响
    if url_part_prefix not in img_url:
        img_url = 'http://img3.img.cn/dwdfs/' + img_url
    return img_url

def modifyBox(box, img_wh):
    '''
    修改归一化的坐标到绝对值坐标
    :param box: 归一化的位置
    :param img_wh: 图片的宽高
    :return:
    '''
    np_box = np.array(box)
    np_box[0::2] = np_box[0::2] * img_wh[0]
    np_box[1::2] = np_box[1::2] * img_wh[1]
    np_box = np_box.astype(np.uint64)
    return np_box.tolist()

def getOneCarRelativeCoord(boxes, img_wh, model='best'):
    '''
    :param boxes: 位置集[0,1]
    :param img_wh: 原图的宽高
    :param model: best/center 面积最大/中心位置的车
    :return:
    '''
    prob_box = 0
    area_box = 0
    dis_center = float('inf')
    flag_center = False  #是否有超过60像素的边
    confidence = 0

    for info in boxes:
        _prob = info['prob']
        _box = info['position']

        w_box = _box[2] - _box[0]
        h_box = _box[3] - _box[1]

        if model=='best':
            area = w_box * h_box
            if (area > area_box) and ((_prob + 0.5) > prob_box):
                area_box = area
                prob_box = _prob
                box = _box
                confidence = _prob

        if model == 'center':
            center_x = (_box[0] + _box[2])/2
            center_y = (_box[1] + _box[3])/2

            flag_box_wh = ((int(img_wh[0] * w_box) < 60) or (int(img_wh[1] * h_box) < 60))
            if flag_box_wh:
                continue

            flag_center = True
            dis = (pow(center_x - 0.5, 2) + pow(center_y - 0.5, 2))
            if  dis < dis_center:
                dis_center = dis
                box = _box
                confidence = _prob

    if (model == 'center') and (not flag_center):
        box = boxes[0]['position']   #中心车辆模式下，若边都小于60像素，则默认为置信度最大的框
        confidence = boxes[0]['prob']
    return box, confidence

def getOneCar(boxes, img_wh, model='best'):
    box, conf = getOneCarRelativeCoord(boxes, img_wh, model)
    box = modifyBox(box, img_wh)
    return box

def cropCarbyPadding(img, rect):
    '''
    rect的位置用于识别，检测时在原图上外扩部分区域
    :param img:
    :param rect:
    :return:
    '''
    img_hw = img.shape[:2]
    box = modifyBox(rect, img_hw[::-1])

    x1, y1, x2, y2 = box
    w_crop = x2 - x1
    h_crop = y2 - y1

    car_crop = img[y1:y2, x1:x2,:]
    x1 = max(0, x1 - int(w_crop*0.3))
    y1 = max(0, y1 - int(h_crop*0.3))
    x2 = min(img_hw[1], x2 + int(w_crop*0.3))
    y2 = min(img_hw[0], y2 + int(h_crop*0.3))
    return img[y1:y2, x1:x2,:], rect

if __name__ == '__main__':
    img_path = '../imgs/1.8.5062.jpg'
    img_buffs = open(img_path, 'rb').read()
    img = bytes2cv(img_buffs)
    print (img.shape)
