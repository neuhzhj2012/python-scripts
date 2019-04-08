#encoding: utf-8
"""
生成视频，并保存图片
"""
import cv2
import os
import re
import imageio
import numpy as np
from moviepy.editor import VideoFileClip

class VideoMaker():
    def __init__(self):
        self.__codec__ = cv2.VideoWriter_fourcc('M','J','P','G')
        self.__codec__ = cv2.VideoWriter_fourcc('m','p','4','v') #视频编码格式
        self.__fps__ = 25
        self.__width__ = 100
        self.__height__ = 80
        self.__dstPath__='data'

        pass
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    def __alphanum_key__(self, s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [self.tryint(c) for c in re.split('([0-9]+)', s)]

    def createVideoFromFiles(self, imageFolder, videoName='video.avi', imgSuffix='.jpg'):
        images = [img for img in os.listdir(imageFolder) if img.endswith(imgSuffix)]

        images.sort(key=self.__alphanum_key__) #按照序号大小重新排序
        frame = cv2.imread(os.path.join(imageFolder, images[0]))
        height, width, layers = frame.shape
        fps = self.__fps__
        video = cv2.VideoWriter(videoName, self.__codec__, fps, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(imageFolder, image)))

        cv2.destroyAllWindows()
        video.release()

    def createVideoFromCamera(self, imageFolder='data', videoName='video.avi'):
        # 创建显示视频的窗口
        cv2.namedWindow('Video')

        # 打开摄像头
        video_capture = cv2.VideoCapture(0)

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width= int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 创建视频写入对象
        video_writer = cv2.VideoWriter(videoName, self.__codec__,fps,(width, height))

        flag_save=True
        img_idx = 0
        if flag_save:
            if not os.path.exists(imageFolder):
                os.makedirs(imageFolder)

        # 读取视频帧，对视频帧进行高斯模糊，然后写入文件并在窗口显示
        success, frame = video_capture.read()
        while success and not cv2.waitKey(1) == 27:
            blur_frame = cv2.GaussianBlur(frame, (3, 3), 0)
            video_writer.write(blur_frame)
            cv2.imshow("Video", blur_frame)
            if flag_save:
                cv2.imwrite(os.path.join(imageFolder, str(img_idx) + ".jpg"), blur_frame)
                img_idx +=1
            success, frame = video_capture.read()

            # 回收资源
        cv2.destroyWindow('Video')
        video_capture.release()

    def createGif(self, imageFolder, gifName='mygif.gif', imgSuffix='.jpg'):
        images_for_gif = []
        images = [img for img in os.listdir(imageFolder) if img.endswith(imgSuffix)]
        for img in images:
            images_for_gif.append(imageio.imread(os.path.join(imageFolder, img)))
        imageio.mimsave(os.path.join(self.__dstPath__, gifName), images_for_gif)

    def releaseVideo(self, filename='video.avi', dst_folder=''):
        vidcap = cv2.VideoCapture(filename)
        success, image = vidcap.read()
        count = 0
        while success:
            rst_path = os.path.join(dst_folder,"frame%d.jpg")
            cv2.imwrite(rst_path%(count), image)  # save frame as JPEG file
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1

class VideoCheck():
    def __init__(self):
        self.__cap_prop_fps__ = cv2.CAP_PROP_FPS
        self.__cap_prop_frame_count__ = cv2.CAP_PROP_FRAME_COUNT
        self.__file_name__=None
        self.__file_obj__ = None
        pass

    def getFileSize(self, filename):
        u"""
        获取文件大小（M: 兆）
        """
        file_byte = os.path.getsize(filename)
        return self.__sizeConvert__(file_byte)

    def getFileTimes(self, filename):
        u"""
        获取视频时长（s:秒）
        """
        clip = VideoFileClip(filename)
        return self.__timeConvert__(clip.duration)

    def getFileFPS(self,filename):
        video = self.get_fileObj(filename)
        fps = video.get(self.__cap_prop_fps__)
        return fps

    def getFileCount(self,filename):
        video = self.get_fileObj(filename)
        count = video.get(self.__cap_prop_frame_count__)
        return count

    def get_fileObj(self, filename):
        if filename != self.__file_name__:
            self.__file_name__=filename
            self.__file_obj__ = cv2.VideoCapture(filename)
        video = self.__file_obj__
        return video

    def __sizeConvert__(self, size):  # 单位换算
        K, M, G = 1024, 1024 ** 2, 1024 ** 3
        if size >= G:
            return str(size / G) + 'G Bytes'
        elif size >= M:
            return str(size / M) + 'M Bytes'
        elif size >= K:
            return str(size / K) + 'K Bytes'
        else:
            return str(size) + 'Bytes'

    def __timeConvert__(self, size):  # 单位换算
        M, H = 60, 60 ** 2
        if size < M:
            return str(size) + u'秒'
        if size < H:
            return u'%s分钟%s秒' % (int(size / M), int(size % M))
        else:
            hour = int(size / H)
            mine = int(size % H / M)
            second = int(size % H % M)
            tim_srt = u'%s小时%s分钟%s秒' % (hour, mine, second)
            return tim_srt

# def save():
#     os.system("ffmpeg -r 1 -i img%01d.png -vcodec mpeg4 -y movie.mp4")

if __name__=='__main__':
    maker = VideoMaker()
    # maker.createVideoFromCamera()
    maker.createGif('data\\tmp')
    # releaseVideo()
    # 视频信息查看
    # Info = VideoCheck()
    # name='video.avi'
    # print 'size:',Info.getFileSize(name)
    # print 'time:',Info.getFileTimes(name)
    # print 'fps:',Info.getFileFPS(name)
    # print 'count:',Info.getFileCount(name)
