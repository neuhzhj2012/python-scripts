#encoding: utf-8
'''
类别的准确率和召回率曲线
文件内容 name label score
'''
import argparse
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


def draw(y_true, y_scores):
    '''
    :param y_true: N*classes，N表示样本数，classes表示类别数，每行数据为one-hot向量
    :param y_scores: N*classes,表示所有样本预测的置信度
    :return:
    '''
    precision, recall, thresholds = precision_recall_curve(y_true.ravel(), y_scores.ravel())
    average_precision = average_precision_score(y_true, y_scores,
                                                average="micro") #平均准确率
    precision = np.array(precision[:-1][::-1]) #降序排列
    recall = np.array(recall[:-1][::-1]) #升序排列
    thresholds = np.array(thresholds[::-1]) #降序排列
    # np.array(precision)>0.99
    print(f'precision_head: {precision[:-1][:15]}\n precision_tail: {precision[:-1][-15:]}\n recall_head: {recall[:-1][:15]}\n recall_tail: {recall[:-1][-15:]}\n thresholds_head: {thresholds[:15]}\n thresholds_tail: {thresholds[-15:]}')

    for i in [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.05, 0]:
        if ((precision>i).tolist().count(1)==0):
            print(f'conf: {i}\tP: 0\tR:0\tthresh:0')
            continue
        idx = precision.tolist().index(precision[precision>i][-1])
        print(f'conf: {i}\tP: {precision[idx]}\tR:{recall[idx]}\tthresh:{thresholds[idx]}')


    plt.figure()
    plt.step(recall, precision, where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision))
    plt.savefig('pr.jpg')

def getArgs():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', help='input img',
                      default='1.jpg')
    return vars(args.parse_args())

if __name__=='__main__':
    classes = ['ad_ocr', 'gun', 'knife', 'normal_ocr', 'normal_people', 'normal_thing', 'qr_code']
    classes = ['4739', '0'] #二维码
    classes = ['1', '0']  #涉政类

    args = getArgs()
    pred_rst_path = args['input']

    buffs = open(pred_rst_path, 'r').readlines()
    lines = [line.strip() for line in buffs]
    print('nums: {}'.format(len(lines)))

    y_gt = list()
    y_score = list()

    # #多分类情况
    # for idx,line in enumerate(lines):
    #
    #     _, label, confs = line.split('\t')
    #
    #     one_hot = len(classes) * [0]
    #     one_hot[classes.index(label)] = 1
    #     y_gt.append(one_hot)
    #     y_score.append(eval(confs))
    #     # y_score.append(float(confs))
    #     # print(idx, y_gt, y_score)

    #二分类情况
    for idx,line in enumerate(lines):

        _, label, confs = line.split('\t')

        one_hot = 0
        if classes[0]==label:
            one_hot=1
        y_gt.append(one_hot)
        y_score.append(eval(confs))
        # y_score.append(float(confs))
        # print(idx, y_gt, y_score)

    draw(np.array(y_gt), np.array(y_score))
