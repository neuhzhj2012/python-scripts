#encoding: utf-8
import os
from xml.dom.minidom import Document
from lxml import etree, objectify
import xml.etree.ElementTree as ET

class XML():
    def __init__(self):
        pass
    def writeXmlRoot(self, anno):
        #写基础字段
        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(    #根目录
            E.folder(anno['src']),        #根目录内容
            E.filename(anno['file_name']),
            E.size(
                E.width(anno['width']),  #子目录内容
                E.height(anno['height']),
                E.depth(anno['channel'])
            ),
        )
        return anno_tree

    def writeXmlSubRoot(self, anno, bbox_type='xyxy'):
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

    def parseXml(self, xml_dst_dir, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        basic_info = dict()

        #查询单个字段及其子字段
        filename = root.find('filename').text #filename
        #size
        img_w = root.find('size').find('width').text #width
        img_h = root.find('size').find('height').text #height
        depth = root.find('size').find('depth').text #depth
        basic_info['src'] = 'autohome'
        basic_info['file_name'] = filename
        basic_info['width'] = img_w
        basic_info['height'] = img_h
        basic_info['channel'] = 3

        anno_tree = self.writeXmlRoot(basic_info)
        print ('anno_tree_type: {}'.format(type(anno_tree)))

        #查询多个字段
        boxes = root.findall('object')
        objects = list()

        flag_valid_classid = False
        for box_info in boxes:
            class_name = box_info.find('name').text
            box = box_info.find('bndbox')
            xmin = box.find('xmin').text  #object子字段的内容
            xmax = box.find('xmax').text
            ymin = box.find('ymin').text
            ymax = box.find('ymax').text

            object = dict()
            object['class_name']=class_name
            object['bndbox']=list()

            object['bndbox']=[xmin, ymin, xmax, ymax]

            anno_tree.append(self.writeXmlSubRoot(object, bbox_type='xyxy')) #处理相同字段不同内容
        xml_name = os.path.join(xml_dst_dir, filename.replace('.jpg', '.xml'))
        print 'dstxml: {}, anno: {}'.format(xml_name, anno_tree)
        self.write(anno_tree, xml_name)
    def write(self, anno_tree, xml_name):
        etree.ElementTree(anno_tree).write(xml_name, pretty_print=True) #写xml文件

    def writeXml(self):
        doc = Document()
        people = doc.createElement("people")
        aperson = doc.createElement("person")
        name = doc.createElement("name")
        personname = doc.createTextNode("Annie")

        doc.appendChild(people)
        people.appendChild(aperson)
        aperson.appendChild(name)
        name.appendChild(personname)
        filename = "people.xml"
        f = open(filename, "w")
        f.write(doc.toprettyxml(indent="  "))
        f.close()


if __name__=='__main__':
    xmlObj = XML()
    xml_src_folder = 'data/' #east数据格式
    xml_dst_folder = 'data\\tmp' #
    xmls = [ file for file in os.listdir(xml_src_folder) if file.endswith('.xml')]

    xmlObj.writeXml()

    cvt_xml_folder = '' #east数据格式转换为voc格式
    for xml_one in xmls:
        xml_path = os.path.join(xml_src_folder, xml_one)
        xmlObj.parseXml(xml_dst_folder, xml_path)
