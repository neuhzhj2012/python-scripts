# encoding: utf-8

import sys
import os
import argparse

def get_all_file(img_folder):
    rtn_list = list()
    for folder, _, imgs in os.walk(img_folder):
        for img in imgs:
            rtn_list.append(os.path.join(folder, img))
    return rtn_list

def param_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False,
        help="path to input image")
    ap.add_argument("-d", "--indir", required=False,
        help="path to image folder")

    ap.add_argument('-s',"--size",dest='imgSize',required=False,type=int,nargs=2,help='img size for model input')
    ap.add_argument('-p', '--port', dest='svcPort',
        help='service port', type=int, default=29977)
    args = ap.parse_args() #Namespace(image=None, imgSize=None, indir=None, svcPort=29977) args.svcPort
    args = vars(ap.parse_args()) #{'svcPort': 29977, 'image': None, 'imgSize': None, 'indir': None}
    return args

def parse_args2(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    print(vars(parser.parse_args(args)))
    return (parser.parse_args(args))

if __name__ == '__main__':
    #基于sys.argv的参数解析 python *.py 0.jpg 'data' 50 100
    # print ('argv: ',sys.argv)
    # assert len(sys.argv) != 1, ('check code for usage') # 由于没有argparse模块，故无法获得参数信息
    #
    # img_path = sys.argv[1]
    # img_folder = sys.argv[2]
    # imgw = sys.argv[3]
    # imgh = sys.argv[4]
    # print ('imgw {}, imgh {}'.format(imgw, imgh))

    #基于argparse模块的参数解析 python *.py -i 1.jpg -s 224 448 --port 7000
    args = param_args()
    if args['image']==None and args['indir']==None and args['imgSize']==None:
        os.system('python {} -h'.format(__file__))
        # os.system('python {} --help'.format(__file__))
        sys.exit()

    img_paths = [args['image']] if args['image']!=None else get_all_file(args['indir'])
    imgWH = args['imgSize']
    print ('imgWH: ', imgWH)

    for img_path in img_paths:
        print ('name: {}'.format(os.path.basename(img_path)))




