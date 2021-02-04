'''
基于dali加载数据和focusloss作为损失的pytorch训练代码
python3 train_dali_log.py -d data/AD -c 4 -b 32 -t 3 -n 75 -f -fa -a "[0.0745787, 0.680225, 0.070105, 0.175092]"
 -d表示数据存放目录,二级结构，train/val； classA, clasB
 -c 表示类别数量
 -b batchsize
 -t num workers of loading data
 -f 是否使用focus loss
 -fa 是否对focus loss增加样本权重
 -a  focus loss的样本权重，顺序为ls时查看到的目录顺序

'''
from __future__ import print_function, division

import os
import time
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from datetime import datetime
from logger import Logger

from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()  # interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import nvidia.dali as dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator


def random_transform(index):
    dst_cx, dst_cy = (200,200)
    src_cx, src_cy = (200,200)

    # This function uses homogeneous coordinates - hence, 3x3 matrix

    # translate output coordinates to center defined by (dst_cx, dst_cy)
    t1 = np.array([[1, 0, -dst_cx],
                   [0, 1, -dst_cy],
                   [0, 0, 1]])
    def u():
        return np.random.uniform(-0.5, 0.5)

    # apply a randomized affine transform - uniform scaling + some random distortion
    m = np.array([
        [1 + u(),     u(),  0],
        [    u(), 1 + u(),  0],
        [      0,       0,  1]])

    # translate input coordinates to center (src_cx, src_cy)
    t2 = np.array([[1, 0, src_cx],
                   [0, 1, src_cy],
                   [0, 0, 1]])

    # combine the transforms
    m = (np.matmul(t2, np.matmul(m, t1)))

    # remove the last row; it's not used by affine transform
    return m[0:2,0:3]


def gen_transforms(batch_size, single_transform_fn):
    out = np.zeros([batch_size, 2, 3])
    for i in range(batch_size):
        out[i,:,:] = single_transform_fn(i)
    return out.astype(np.float32)

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,size=256, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        #self.res = ops.Resize(device="gpu", resize_x=size, resize_y=size, interp_type=types.INTERP_LINEAR)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_LINEAR)
        self.rescrop = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.bc = ops.BrightnessContrast(device="gpu",brightness=0.5, contrast=0.6)

        # Will flip vertically with prob of 0.1
        self.vert_flip = ops.Flip(device='gpu', horizontal=0)
        self.vert_coin = ops.CoinFlip(probability=0.4)
        
        self.transform_source = ops.ExternalSource()
        self.warp_keep_size = ops.WarpAffine(
            device = "gpu",
          # size                              # keep original canvas size
            interp_type = types.INTERP_LINEAR # use linear interpolation
        )

        # My workaround for Dali not supporting random affine transforms: 
        # a "synthetic random" warp affine transform.
        
        # Rotate within a narrow range with probability of 0.075
        self.rotate = ops.Rotate(device='gpu')
        self.rotate_range = ops.Uniform(range = (-20.0, 20.0))
        self.rotate_coin = ops.CoinFlip(probability=0.075)

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()

        prob_vert_flip = self.vert_coin()
        prob_rotate = self.rotate_coin()
        angle_range = self.rotate_range()
        self.transform = self.transform_source()

        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        ##images = self.vert_flip(images, vertical=prob_vert_flip) # Specify prob_vert_flip here 
        images = self.rotate(images, angle=angle_range)
        images = self.warp_keep_size(images.gpu(), matrix = self.transform.gpu())
        images = self.res(images)
        images = self.rescrop(images)

        
        #images = self.bc(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels.gpu()]

    # Since we're using ExternalSource, we need to feed the externally provided data to the pipeline

    def iter_setup(self):
        # Generate the transforms for the batch and feed them to the ExternalSource
        self.feed_input(self.transform, gen_transforms(self.batch_size, random_transform))


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                           world_size=1,
                           local_rank=0):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                    data_dir=image_dir + '/train',
                                    crop=crop, world_size=world_size, local_rank=local_rank)
        pip_train.build()
        #dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size)
        dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size,auto_reset=True)
        return dali_iter_train
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                data_dir=image_dir + '/val',
                                crop=crop, size=val_size, world_size=world_size, local_rank=local_rank)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(pip_val, size=pip_val.epoch_size("Reader") // world_size, auto_reset=True)
        return dali_iter_val




def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

class DATAPREFEACHER():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()
    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    num = (y_pred_tag == y_test).sum().float()
    
    return num

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, **kwargs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loader = get_imagenet_iter_dali(type='train', image_dir=data_dir, batch_size=batch_size,
                                          num_threads=num_workers, crop=224, device_id=0, num_gpus=1)
    val_loader = get_imagenet_iter_dali(type='val', image_dir=data_dir, batch_size=batch_size,
                                          num_threads=num_workers, crop=224, device_id=0, num_gpus=1)
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        logger.append('Epoch {}/{}'.format(epoch, num_epochs - 1))           
        logger.append('-' * 10)

        #train_loader = get_imagenet_iter_dali(type='train', image_dir=data_dir, batch_size=batch_size,
        #                                  num_threads=num_workers, crop=224, device_id=0, num_gpus=1)
        #val_loader = get_imagenet_iter_dali(type='val', image_dir=data_dir, batch_size=batch_size,
        #                                  num_threads=num_workers, crop=224, device_id=0, num_gpus=1)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            dataset_sizes = 0

                # Iterate over data.
            tm_start = time.time()
            for i, data in enumerate(dataloader):
                inputs = data[0]["data"].cuda(non_blocking=True)
                labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                    
                flag_bce = False
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    #preds=outputs
                    #outputs = outputs.squeeze()
                    #print(outputs)
                    if flag_bce:
                        loss = criterion(outputs, labels.unsqueeze(1).type_as(outputs))
                    else:
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if flag_bce:
                    running_corrects += binary_acc(outputs, labels)
                else:
                    running_corrects += torch.sum(preds == labels.data)
                dataset_sizes += inputs.size(0) #图片总数


            tm_epoch = time.time() - tm_start
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects.double() / dataset_sizes

            print('{} Loss: {:.4f} Acc: {:.4f} tm: {}'.format(
                phase, epoch_loss, epoch_acc, tm_epoch))
            logger.append('{} Loss: {:.4f} Acc: {:.4f} tm: {}'.format(
                phase, epoch_loss, epoch_acc, tm_epoch))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch % 10 == 0 and epoch > 0):
            torch.save(best_model_wts, 'resnet_ft_{}.pth'.format(epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))
    logger.append('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.append('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs = data[0]["data"].cuda(non_blocking=True)
            labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(preds[j]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def getArgs():
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--datadir', help='img folder containing train & val data',
                      default='data')  #数据目录，包含train，val两个目录，
    args.add_argument('-c', '--class_nums', help='labels num',type=int,
                      default='1000')
    args.add_argument('-n', '--nepochs', help='train epochs',type=int,
                      default='30')
    args.add_argument('-l', '--learning_ratio', help='init learning ratio',type=float,
                      default='0.01')
    args.add_argument('-b', '--batch_size', help='batch size',type=int,
                      default='16')
    args.add_argument('-t', '--read_thread', help='num thread for reading data',type=int,
                      default='1') #加载数据的线程数
    args.add_argument('-a', '--alpha_ratio', help='ratio of sample nums',
                      default='None') #focalloss中样本权重的alpha参数，eg:[0.1,0.3,0.6]

    args.add_argument('-p', '--prefeacher', help='data prefeacher',
                      action='store_true')  #是否使用数据预加载
    args.add_argument('-s', '--is_show', help='show train data', action='store_true') #是否展示图片，仅可视化用
    args.add_argument('-f', '--is_focalloss', action='store_true') #是否使用focal loss作为损失函数
    args.add_argument('-fa', '--is_focalloss_alpha', action='store_true') #focal loss中是否使用alpha参数
    return vars(args.parse_args())

if __name__ == '__main__':
    args = getArgs()
    data_dir = args['datadir'] #数据目录
    class_nums = args['class_nums']
    nepochs = args['nepochs']  #训练epochs
    learning_rate = args['learning_ratio'] #学习率
    batch_size = args['batch_size']
    num_workers = args['read_thread']
    alpha_ratio = args['alpha_ratio'] #训练样本所占的比例信息，

    is_prefeacher = args['prefeacher'] #是否进行数据预加载来提升训练速度
    is_show = args['is_show'] #是否查看增强后的训练数据
    is_focalloss = args['is_focalloss'] #是否使用focal-loss
    is_focalloss_alpha = args['is_focalloss_alpha'] #在focal-loss基础上增加对样本不均衡的处理

    print('datadir: {}, nepochs: {}, lr: {}, bz: {}, num_workers: {}, alpha_ratio: {}, is_prefetch: {},is_focalloss: {}, is_focalloss_alpha: {}'
          .format(data_dir, nepochs, learning_rate, batch_size, num_workers, alpha_ratio, is_prefeacher,is_focalloss, is_focalloss_alpha))


    now=datetime.now()
    tm_now=now.strftime("%m%d%H")

    data_subfolder = data_dir.split(os.path.sep)[-1] if data_dir[-1] != os.path.sep else data_dir.split(os.path.sep)[-2]
    logger = Logger(os.path.join('trainlog', data_subfolder+'_'+tm_now+'.log'))

    logger.append(args)
    train_loader = get_imagenet_iter_dali(type='train', image_dir=data_dir, batch_size=batch_size,
                                          num_threads=num_workers, crop=224, device_id=0, num_gpus=1)
    val_loader = get_imagenet_iter_dali(type='val', image_dir=data_dir, batch_size=batch_size,
                                          num_threads=num_workers, crop=224, device_id=0, num_gpus=1)

    #model parameters
    model_ft = models.resnet50(pretrained=True)
    #model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, class_nums) #迁移学习，更新输出类别数
  
    #model_ft = models.mobilenet_v2(pretrained=True)
    #model_ft.classifier[1] = nn.Linear(model_ft.last_channel, class_nums)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    #criterion = torch.nn.BCEWithLogitsLoss()

    # https://zhuanlan.zhihu.com/p/28527749
    # focal-loss
    if is_focalloss:
        if is_focalloss_alpha:
            assert alpha_ratio != 'None'
            # alpha_ratio = [0.089955, 0.125509, 0.131806, 0.12765, 0.094667, 0.089955, 0.0836153, 0.0308417, 0.176141, 0.0498608]
            alpha_ratio = eval(alpha_ratio)
            alpha_ratio_tensor = torch.FloatTensor(alpha_ratio)
        else:
            alpha_ratio_tensor=None
        print('##in focalloss##')
        criterion = FocalLoss(class_nums, alpha=alpha_ratio_tensor)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    #训练:
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=nepochs)
    
    data_subfolder = data_dir.split(os.path.sep)[-1] if data_dir[-1] != os.path.sep else data_dir.split(os.path.sep)[-2]

    dst_model_path = 'resnet_ft_{}.pth'.format(data_subfolder + '-'+str(class_nums)+'_'+tm_now)
    torch.save(model_ft.state_dict(), dst_model_path)
    print('savemodel: {}'.format(dst_model_path))
    logger.append('savemodel: {}'.format(dst_model_path))
