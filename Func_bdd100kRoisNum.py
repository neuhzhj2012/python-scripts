#encoding: utf-8
'''
bdd100k检测集中共12个类别：{'traffic light', 'train', 'drivable area', 'traffic sign', 'lane',
                            'person', 'bus', 'car', 'truck', 'rider', 'bike', 'motor'}
其中(以训练集中0000f77c-6257be58.jpg为例)：
drivable area为点集{'category': 'drivable area', 'attributes': {'areaType': 'direct'}, 'manualShape': True, 'manualAttributes': True, 'poly2d': [{'vertices': [[1280.195648, 626.372529], [1280.195648, 371.830705], [927.081254, 366.839689], [872.180076, 427.979637], [658.814135, 450.439209], [585.196646, 426.731883], [0, 517.817928], [0, 602.665203], [497.853863, 540.2775], [927.081254, 571.471352], [1280.195648, 626.372529]], 'types': 'LLLLLLLLCCC', 'closed': True}], 'id': 7}
lane为线{'category': 'lane', 'attributes': {'laneDirection': 'parallel', 'laneStyle': 'solid', 'laneType': 'road curb'}, 'manualShape': True, 'manualAttributes': True, 'poly2d': [{'vertices': [[503.674413, 373.137193], [357.797732, 374.672737]], 'types': 'LL', 'closed': False}], 'id': 8}
其他均为矩形框
{'category': 'traffic light', 'attributes': {'occluded': False, 'truncated': False, 'trafficLightColor': 'green'}, 'manualShape': True, 'manualAttributes': True, 'box2d': {'x1': 1125.902264, 'y1': 133.184488, 'x2': 1156.978645, 'y2': 210.875445}, 'id': 0}
{'category': 'car', 'attributes': {'occluded': False, 'truncated': False, 'trafficLightColor': 'none'}, 'manualShape': True, 'manualAttributes': True, 'box2d': {'x1': 45.240919, 'y1': 254.530367, 'x2': 357.805838, 'y2': 487.906215}, 'id': 4}
'''
import json

class_rois=['car', 'bus', 'truck', 'person', 'rider', 'bike', 'motor']

train_labels= 'bdd100k_labels_images_train.json'
val_labels= 'bdd100k_labels_images_val.json'

with open(train_labels) as fp:
    data = json.load(fp)

roi_name = dict() #统计不同类别的图片名
rois = dict() #统计不同类别的物体数
for roi in class_rois:
    roi_name[roi]=list()
    rois[roi]=list()

for info in data:
    name=info['name']
    attrs_time = info['attributes']['timeofday'] #daytime, night
    for roi_info in info['labels']:
        # print (roi_info)
        label = roi_info['category']
        if 'box2d' in roi_info.keys():
            xxyy = list(roi_info['box2d'].values())
        if (label in class_rois) :
            rois[label].append(name) #统计每个类别的数量
            if (name not in roi_name[label]):
                roi_name[label].append(name) #统计当前类别的图片数

for label in class_rois:
    print ('{} img_num: {}, roi_num: {}'.format(label, len(roi_name[label]), len(rois[label])))




