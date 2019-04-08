#encoding: utf-8
import cv2
import os
import re
import math
import shutil
import random
import imageio
import numpy as np
from PIL import Image
import imagehash
from sklearn.metrics.cluster import entropy
from skimage.measure import compare_ssim as ssim

class IMGLIKE():
    def __init__(self):
        '''
        图片相似度比较方法
        https://blog.csdn.net/jacke121/article/details/81154252
        图片相似度比较compare_ssim 对光照变化敏感，1ms
        phash 哈希感知，图像光照变化不受影响，2ms，图像内容不一样，也会比较得到相同的值
        https://testerhome.com/topics/16287
        直方图比较
        峰值信噪比PSNR
        SSIM在图像去噪中、图像相似度评价上全面超越MSE、SNR、PSNR等方法
        '''
        pass
    def cmpEntropy(self, img1, img2): #图片熵比较，取值范围[0，无穷）
        return abs(entropy(img1) - entropy(img2))  # 关键帧相邻两图片的熵值

    def cmpEuclidean(self, img1, img2):
        #对应像素的欧式距离比较,不具有实际应用意义,取值范围[0,无穷）
        return cv2.norm(img1, img2, cv2.NORM_L2)

    def cmpSSIM(self, img1, img2):
        #对相同分辨率的图进行结构相似性评价，表示失真程度，值越大，失真越小， 取值范围[0,1]
        #0.5作为失真的阈值，即两张图变化较大
        return ssim(img1, img2, multichannel=True)

    def cmpHash(self, img1, img2,imgsize=8, mode=0):
        '''
        归一化后值的取值范围为[0,1]，值越大，越相近，0.9作为阈值，小于0.9则变化较大
        :param img1:
        :param img2:
        :param imgsize 哈希窗口
        :param mode 哈希类型：0表示差值哈希，1表示感知哈希
        :return:
        '''
        func_dict = {0:self.getDHash, 1:self.getPHash} #dhash,差值哈希，phash,感知哈希
        func = func_dict[int(mode)]

        hash1 = func(img1, imgsize)
        hash2 = func(img2, imgsize)
        # hash长度不同则返回-1代表传参出错
        dis_hamming = self.hamming_distance(hash1, hash2)
        return 1 - float(dis_hamming) / (imgsize * imgsize)

    def cmpHist(self, img1, img2):
        '''
        原文中为图像分区域后计算的
        :param img1:
        :param img2:
        :return:
        '''
        img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        img1_hist = img1_pil.histogram()
        img2_hist = img2_pil.histogram()
        assert len(img1_hist) == len(img2_hist), "error"

        data = []

        for index in range(0, len(img1_hist)):
            if img1_hist[index] != img2_hist[index]:
                data.append(1 - abs(img1_hist[index] - img2_hist[index]) / max(img1_hist[index], img2_hist[index]))
            else:
                data.append(1)

        return sum(data) / len(img1_hist)

        pass

    def getSimilarity(self, imgs, mode='0'):
            # func_dict = {0:self.cmpEntropy, 1:self.cmpEuclidean, 2:self.cmpSSIM, 3:self.cmpPhash, 4:self.cmpHist}
            #像素的均方差不具有应用价值
            func_dict = {0:self.cmpEntropy,1:self.cmpSSIM, 2:self.cmpHash, 3:self.cmpHist}
            func = func_dict[int(mode)]
            similarity_sum = 0
            for i in range(len(imgs)-1):
                im1 = cv2.imread(imgs[i], 0)
                im2 = cv2.imread(imgs[i + 1], 0)
                similarity = func(im1,im2)
                print 'model: {}, similarity: {}'.format(int(mode), similarity)
                similarity_sum += similarity
            return similarity_sum / (len(imgs) - 1)

    def cvtCv2Image(self, img):
        if  len(img.shape)==3:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            image = Image.fromarray(img)
        return image

    def getPHash(self, image, hashSize=8):
        image = self.cvtCv2Image(image)
        image_hash = imagehash.phash(image, hash_size=hashSize)
        return str(image_hash)
    def getDHash(self, image, hashSize=8):
        '''
        dhash,差值哈希
        :param image:
        :param hashSize:
        :return:
        '''
        # resize the input image, adding a single column (width) so we
        # can compute the horizontal gradient

        #基础实现逻辑
        # if len(image.shape) == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #
        # resized = cv2.resize(image, (hashSize + 1, hashSize))
        #
        # # compute the (relative) horizontal gradient between adjacent
        # # column pixels
        # diff = resized[:, 1:] > resized[:, :-1]
        #
        # # convert the difference image to a hash
        # return str(sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v]))
        image = self.cvtCv2Image(image)
        image_hash = imagehash.dhash(image, hash_size=hashSize)
        return str(image_hash)

    def hamming_distance(self, s1, s2):
        """Return the Hamming distance between equal-length sequences"""
        if len(s1) != len(s2):
            raise ValueError("Undefined for sequences of unequal length")
        return sum(el1 != el2 for el1, el2 in zip(s1, s2))

class SAMPLE():
    def __init__(self):
        '''
        视频中采样的方法
        '''
        self.total_num = 100 #每段视频的图片数
        self.sample_num = 3 #每段视频的关键帧数目
        self.iter = 2 #算法迭代次数
        self.sample_set_num = 6 #候选关键帧的组合数
        self.inpath = "data"
        self.output_path = os.path.join(self.inpath, 'rst')
        self.sample_set = []  # Population matrix. #候选帧组合的列表 + 当前组合的相似度
        self.MV = []  # Mutation vector. #修改每组关键帧的组合列表
        self.TV = []  # Trail vector.
        self.F = 0.9  # Scale factor.
        self.Cr = 0.6  # Cr probability value.
        self.similarity=IMGLIKE()
        pass

    def tryint(self, s):
        try:
            return int(s)
        except ValueError:
            return s

    def __alphanum_key__(self, s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [self.tryint(c) for c in re.split('([0-9]+)', s)]

    def randomSample(self, imgs):
        '''
        初始化图片组合
        :return:
        '''
        for i in range(self.sample_set_num):
            # sample = sorted(random.sample(range(1, self.total_num+1), self.sample_num))
            sample_num = min(len(imgs), self.sample_num)
            sample = random.sample(imgs, self.sample_num )
            sample.sort(key=self.__alphanum_key__)
            self.sample_set.append(sample) #随机抽取TOTAL_KEY_FRAMES作为关键帧
            model = i%4
            self.sample_set[-1].append(self.similarity.getSimilarity(self.sample_set[-1], mode=model)) #每个候选集新增一位熵的结果
            print self.sample_set[-1]

# # MUTATION
# def mutation(parent):
#     '''
#     # 根据每个位置与其他组合中对应位置的组合，重置当前位置的帧id
#     感觉此处主要是重置每个位置的id，使其不同组合中的关键帧位置大体相同
#     :param parent:
#     :return:
#     '''
#     R = random.sample(self.sample_set,2)
#     global MV
#     MV[:] = []
#     MV_value = 0
#     # print self.sample_set[parent]
#     for i in range(sample_num):
#         MV_value = int(self.sample_set[parent][i] + F*(R[0][i] - R[1][i]))
#         if(MV_value < 1):
#             MV.append(1)
#         elif(MV_value > total_num):
#             MV.append(total_num)
#         else:
#             MV.append(MV_value)
#     MV.sort()
#     MV.append(getEntropy(MV))
#
# # CROSSOVER (uniform crossover with Cr = 0.6).
# def crossover(parent, mutant):
#     # print "mutant: ", mutant
#     # print "parent: ", parent
#     for j in range(sample_num) :
#         if(random.uniform(0,1) < Cr) :
#             TV.append(mutant[j])
#         else:
#             TV.append(parent[j])
#     TV.sort()
#     TV.append(getEntropy(TV))
#     # print "TV    : ", TV
#
# # SELECTION : Selects offspring / parent based on higher Entropy diff. value.
# def selection(parent, trail_vector):
#     #保留熵大的关键帧组合
#     if(trail_vector[-1] > parent[-1]):
#         parent[:] = trail_vector #变量替换
#         # print "yes", parent
#         print "replace: yes"
#     else:
#         print "replace: no"
#
# # bestParent returns the parent with then maximum Entropy diff. value.
# def bestParent(population):
#     Max_Entropy_value = population[0][-1]
#     Best_Parent_Index = population[0]
#     for parent in population:
#         if (parent[-1] > Max_Entropy_value):
#             Max_Entropy_value = parent[-1]
#             Best_Parent_Index = parent
#     return Best_Parent_Index


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def main_img():
    sample = SAMPLE()
    simi = IMGLIKE()
    # img
    img1_path = 'data\\video\\23.jpg'
    img2_path = 'data\\video\\frame286.jpg'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    hash_size = 8

    print 'entropy: {}'.format(simi.cmpEntropy(img1, img2))  # 越相似则熵值越小
    # print 'euclidean: {}'.format(simi.cmpEuclidean(img1, img2)) #无实际应用意义
    print 'ssim: {}'.format(simi.cmpSSIM(img1, img2))  # 图像分辨率得相同, 0.5以下认为变化较大的帧
    print 'hash: {}'.format(simi.cmpHash(img1, img2, hash_size))  # 0.9以下认为变化较大的帧
    print 'hist: {}'.format(simi.cmpHist(img1, img2))

def main_video():
    sample = SAMPLE()
    simi = IMGLIKE()

    #folder
    img_folder='data\\tmp'
    imgs_name = os.listdir(img_folder)
    imgs_path = [os.path.join(img_folder, name) for name in imgs_name]
    imgs_path.sort(key=alphanum_key)

    hash_size=8
    thresh = 0.9
    for idx in range(len(imgs_path) - 1):
        img1 = cv2.imread(imgs_path[idx])
        img2 = cv2.imread(imgs_path[idx + 1])

        simi_value = simi.cmpHash(img1, img2, hash_size)
        print "simi_value: {}, name1: {} {}".format(simi_value, os.path.basename(imgs_path[idx]), os.path.basename(imgs_path[idx+1]))
        if simi_value < thresh:
            print '**simi_value: {}, name: {}'.format(simi_value, os.path.basename(imgs_path[idx + 1]))


    # sample.randomSample(imgs_path)

if __name__=='__main__':
    # main_img()
    main_video()