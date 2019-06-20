#encoding: utf-8
'''
图像清晰度评价(模糊判断)
https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
'''
from imutils import paths
import argparse
import cv2
import os


class BlurDet():
    def __init__(self):
        pass

    def BrennerDetection(self, img_gray):
        '''
        无法区分自建数据集中的是否模糊图片
        :param img_gray: 灰度图
        :return:
        '''
        img_h, img_w = img_gray.shape
        score = 0
        for i in range(img_h):
            for j in range(img_w - 2):
                score += (int(img_gray[i, j + 2]) - int(img_gray[i, j])) ** 2
        return score / 100000
    def variance_of_laplacian(self, img_gray):
        '''
        无法区分自建数据集中的是否模糊图片
        :param img_gray: 灰度图
        :return:
        '''
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(img_gray, cv2.CV_64F).var()

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
folder = 'D:\\Data\\Video\\blur'
imgs = os.listdir(folder)
imgs_path = [os.path.join(folder, img) for img in imgs]
# for imagePath in paths.list_images(args["images"]):
for imagePath in imgs_path:
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    image = cv2.imread(imagePath)
    img_rsz_width = 400
    image = cv2.resize(image, (img_rsz_width, image.shape[0] * img_rsz_width/image.shape[1]), interpolation=cv2.INTER_CUBIC) #宽400
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = blurObj.variance_of_laplacian(gray)
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