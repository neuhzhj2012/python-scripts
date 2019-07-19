#encoding: utf-8
'''
图像清晰度评价(模糊判断)
https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
'''
import argparse
import cv2
import os
import numpy as np
import shutil

class BlurDet():
    def __init__(self):
        pass
    def variance_of_laplacian(self, img_gray):
        '''
        无法区分自建数据集中的是否模糊图片
        :param img_gray: 灰度图
        :return:
        '''
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(img_gray, cv2.CV_64F).var()

    def getIllumination(self, img):
        assert img.shape[-1] == 3, 'color image needed!'
        img2hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgValue = img2hsv[:,:,-1]
        mask = imgValue > 0
        print ('mask num: {}'.format(mask.tolist().count(True)))
        return np.mean(imgValue) #全局平均
        # return np.mean(imgValue[mask]) #掩码平均

def getAllFiles(folder, strType='jpg'):
    names = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if strType not in file:
                continue
            names.append(os.path.join(root, str(file)))
    return names

def get_param():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
                    help="path to input directory of images")
    ap.add_argument("-t", "--threshold", type=float, default=100.0,
                    help="focus measures that fall below this value will be considered 'blurry'")
    args = vars(ap.parse_args())
    return args

blurObj = BlurDet()
# loop over the input images
folder = 'D:\\Python\\Jupyter\\data\\sharpness'

imgs = os.listdir(folder)
imgs_path = [os.path.join(folder, img) for img in imgs]

for imagePath in imgs_path:
    image = cv2.imread(imagePath)
    if isinstance(image, type(None)):
        print ('img: {} not exist'.format(os.path.basename(imagePath)))
        continue
    img_rsz_width = 400
    image = cv2.resize(image, (img_rsz_width, int(image.shape[0] * img_rsz_width/image.shape[1])), interpolation=cv2.INTER_CUBIC) #宽400
    image = cv2.resize(image, (img_rsz_width, img_rsz_width), interpolation=cv2.INTER_CUBIC) #宽400

    #亮度判断
    illu_value = blurObj.getIllumination(image)
    if illu_value < 30 or illu_value > 230:
        shutil.copy(imagePath, 'rst/%.2f_%s'%(illu_value, os.path.basename(imagePath)))
        print ('img: {}, illumination: {}'.format(os.path.basename(imagePath), illu_value))
    continue

    #judge blur ,即使相同分辨率也无法判定模糊图片的阈值
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = blurObj.variance_of_laplacian(gray) #laplace无法确定阈值
    text = "Not Blurry"

    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < 100:
        text = "Blurry"

    # show the image
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    key = cv2.waitKey(0)