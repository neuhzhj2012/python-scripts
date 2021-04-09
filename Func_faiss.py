#encoding:utf-8
import cv2,os
import numpy as np
import sys
import shutil
import faiss
import argparse
from PIL import Image
from functools import reduce
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
'''
hash算法：https://zhuanlan.zhihu.com/p/63180171
faiss:一文搞懂faiss计算 https://zhuanlan.zhihu.com/p/133210698
'''
#均值哈希算法
def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str
#差值感知算法
def dHash(img):
    #缩放8*8
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    #转换灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

def avHash(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    # return reduce(lambda x, (y, z):x | (z << y),
    #               enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0)
    return str(reduce(lambda x, y_z : x | y_z[1] << y_z[0], enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0))


model = models.resnet50(pretrained=True)
modules=list(model.children())[:-1]
model=nn.Sequential(*modules)
model.eval()

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def encoding(img):
    try:
      img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

      img = trans(img_pil)
      img = img.unsqueeze(0)
      encod = torch.reshape(model(img)[0], (-1,))[:ndim].detach().numpy()
    except:
      encod = np.array([0])
    return encod


ndim = 2048 #aHash,dHash 64, avHash 19, encoding 2048
nlist = 5 #聚类中心个数
nbatch = 100
nquery = 10

def getImgs(img_folder):
    rst = list()
    for sub_dir, _, names in os.walk(img_folder):
        for name in names:
            if name.split('.')[-1] not in ['jpg', 'jpeg', 'png']:
                continue
            rst.append(os.path.join(sub_dir, name))
    return rst

def getArgs():
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--folder', help='input folder',
                      default='img')
    args.add_argument('-i', '--img', help='search img',
                      default='img/1.jpg')

    return vars(args.parse_args())

if __name__ == '__main__':
    args = getArgs()
    img_folder = args['folder']
    img_q = args['img']
    imgs = getImgs(img_folder)

    fun = encoding
    print('img nums: {}'.format(len(imgs)))

    # index = faiss.IndexFlatL2(ndim) #构建索引
    # print('Train: ',index.is_trained)

    # quantizer = faiss.IndexFlatL2(ndim)
    quantizer = faiss.IndexFlatIP(ndim)
    faiss.index_factory(2, 'Flat', faiss.METRIC_INNER_PRODUCT)
    # index = faiss.IndexIVFFlat(quantizer, ndim, nlist, faiss.METRIC_L2)
    # assert not index.is_trained  # 倒排表索引类型需要训练
    # index.train(img_bank)  # 训练数据集应该与数据库数据集同分布
    # assert index.is_trained
    index = faiss.IndexIDMap(quantizer)

    init = True

    if init:
        img_bank = np.zeros((len(imgs), ndim)).astype('float32')
        # img_bank = list()
        img_id_bank = list()
        name_idx = dict()
        for idx, img_path in enumerate(imgs):
            img_name = os.path.basename(img_path)
            if img_folder[-1]!=os.path.sep:
                img_name = img_path.replace(img_folder+os.path.sep,'')
            else:
                img_name = img_path.replace(img_folder , '')
            # print(idx, os.path.basename(img_path))
            img = cv2.imread(img_path)
            img_encoding_ = fun(img)

            # print('type: {}'.format(type(img_encoding_)))
            if isinstance(img_encoding_, str):
                img_encoding_ = np.array(list(img_encoding_)).astype('float32')
            norm = np.linalg.norm(img_encoding_)

            img_encoding = img_encoding_ / norm

            name_idx[idx]=img_name
            # img_bank[idx,:] = np.array(list(img_encoding)).astype('float32') #对应faiss index.add
            img_bank[idx,:] = np.array(img_encoding).astype('float32') #对应faiss index.add
            # img_bank.append(img_encoding)
            # img_id_bank.append(idx)
            index.add_with_ids(np.array([img_encoding]).astype('float32'), np.array([idx]).astype('int64'))

        print(img_bank.shape if isinstance(img_bank, (np.ndarray, np.generic)) else len(img_bank))
        # index.add(img_bank)  #对应IndexIVFFlat
        print(np.array(img_bank))
        # index.add_with_ids(np.array(img_bank),np.array(img_id_bank))

        faiss.write_index(index, 'faiss_index.index')
        with open('name_id.txt', 'w') as fp:
            fp.write(str(name_idx))
    else:
        index = faiss.read_index('faiss_index.index')
        name_idx = eval(open('name_id.txt', 'r').readline())

    # print(name_idx)

    img_query = np.zeros((1, ndim)).astype('float32')
    # img_query = np.zeros((1, ndim))
    img = cv2.imread(img_q)
    img_encoding_ = fun(img)


    if isinstance(img_encoding_, str):
        img_encoding_ = np.array(list(img_encoding_)).astype('float32')
    norm = np.linalg.norm(img_encoding_)
    img_encoding = img_encoding_ / norm
    print(img_encoding)
    img_query[0,:] = img_encoding.astype('float32') #string to array




    print('index num: {}'.format(index.ntotal))

    def getnames(Idx):
        names = np.zeros((Idx.shape[0], Idx.shape[1])).astype('object') #防止字符串被截断，np.string_默认为32的字符长度
        # print(names.shape, names.dtype)
        for id_row in range(Idx.shape[0]):
            for id_col in range(Idx.shape[1]):
              try:  
                names[id_row, id_col] = name_idx[Idx[id_row,id_col]]
              except:
                print("error")
        return names

    #测试
    # k = 4
    # D, I = index.search(img_bank[:2], k)
    # print ("IIIIIIIIIIII")
    # print (I)
    # print(getnames(I))
    # print ("ddddddddd")
    # print (D)



    k = len(imgs)
    D, I = index.search(img_query, k)
    print ("IIIIIIIIIIII")
    print (I)
    names=getnames(I)[0]
    print(names)
    print ("ddddddddd")
    print (D)
    for name, dis in zip(names,D[0]):
        if dis > 0.75:
            src_img = os.path.join(img_folder, name)
            if os.path.exists(src_img):
                # shutil.move(os.path.join('Imgs/selfie', name), 'Imgs/rst/') #重名时报错
                shutil.move(src_img, os.path.join('rst/', name)) #可覆盖移动
                # shutil.copy(src_img, os.path.join('Imgs/rst/', name))

