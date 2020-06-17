#encoding: utf-8
import io, sys, os
import time
import base64
import asyncio
import requests
import argparse
import simplejson as json
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def getArgs():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', help='input image', default='0.jpg')
    return vars(args.parse_args())

def getAllfiles(folder):
    rtn = list()
    for dirname, _, imgs in os.walk(folder):
        for img in imgs:
            rtn.append(os.path.join(dirname, img))
    return rtn

class CustomError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) #初始化父类
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo

executor = ThreadPoolExecutor(2)

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

class HDFS(object):
    fastdfs_url = None  #上传图片地址
    fastdfs_url_get_token = None #token获取地址
    fastdfs_key = None
    fastdfs_source = 'dmcv'
    tm_token_valid = '1' #有效单位：分钟，默认为24h

    url_prefix = 'https://img/'


    def __init__(self, when='h', interval=1):   #由于token最长有效时间为24小时，且单位为minutes,故when(时间单位)只有h, m,自定义个MIDNIGHT(半夜,需要配合tm_token_valid为24*60小时)
        super().__init__()
        self.payload = {'source': self.fastdfs_source, 'minutes': self.tm_token_valid}
        self.payload['key'] = self.fastdfs_key
        self.token = None

        self.when = when.upper()
        if self.when=='S':
            self.interval = 1
        elif self.when == 'M':
            self.interval = 60 # one minute
        elif self.when == 'H':
            self.interval = 60 * 60 # one hour
        self.interval = self.interval * interval
        self.reset()

    def reset(self):  #更新token
        t = int(time.time())
        self.rolloverAt = self.computeRollover(t)
        self.getToken()

    def computeRollover(self, currentTime):
        result = currentTime + self.interval
        if self.when == 'MIDNIGHT':
            t = time.localtime(currentTime)
            currentHour = t[3]
            currentMinute = t[4]
            currentSecond = t[5]
            # r is the number of seconds left between now and midnight
            r = 24 * 60 * 60 - ((currentHour * 60 + currentMinute) * 60 +
                    currentSecond)
            result = currentTime + r

        return result

    def shouldRollover(self):
        """
        Determine if rollover should occur.
        """
        t = int(time.time())
        if t >= self.rolloverAt:
            return 1
        return 0

    def getToken(self):
        result = requests.post(self.fastdfs_url_get_token, data=self.payload)
        print ('fastdfs tm_post: {}'.format(result.elapsed.total_seconds()))
        result = eval(result.text)
        assert result['code']==0,'get fastdfs token error'
        print (result)
        self.token = result['result']

    async def savetoDfs(self, img_string, extname='jpg'):
        tm_start = time.time()
        if self.shouldRollover():
            # print ('doRollover')
            self.reset()
        tm_end = time.time()
        # print ('tm shouldRollover: {}'.format(tm_end - tm_start))
        payload = {'source': self.fastdfs_source, 'exName': extname, 'token': self.token}
        def fun():
            return requests.post(self.fastdfs_url, files={'file': img_string}, data=payload)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, fun)
        # requests.post(self.fastdfs_url, files={'file': img_string}, data=payload)
        print ('tmdfs: {}, rst: {}'.format(result.elapsed.total_seconds(), eval(result.text)))
        result = eval(result.text)

        if result['code'] != 0:
            self.reset()
            raise CustomError('fastdfs error {}'.format(result)) #主动抛出异常
        return self.url_prefix + result['result']

class AUTOCAR(object):
    host_recog = None
    data = dict()
    def __init__(self):
        pass

    #@asyncio.coroutine
    async def recog(self, img_string):
        self.data['image_base64'] = img_string
        def fun():
            # return requests.post(self.host_recog, json = self.data)
            return session.post(self.host_recog, json = self.data)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, fun)
        # result = requests.post(self.host_recog, json = self.data)
        print('tmrecog: {}, rst: {}'.format(result.elapsed.total_seconds(), eval(result.text)))
        tm = result.elapsed.total_seconds()
        result = eval(result.text)
        return result


if __name__ == '__main__':
    myDfs = HDFS(when='m', interval=1)
    myCar = AUTOCAR()
    img_path = '0.jpg'

    args = getArgs()
    img_path = args['input']

    if os.path.isfile(img_path):
        img_paths = [img_path]
    else:
        img_paths = getAllfiles(img_path)

    loop = asyncio.get_event_loop()
    # for _ in range(2):
    #     img_string = open(img_path,'rb').read()
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        img_string = open(img_path, 'rb').read()
        tm_start=time.time()
        groups = asyncio.gather(myDfs.savetoDfs(img_string), myCar.recog(base64.b64encode(img_string)))
        tm_end = time.time()
        print ('tmgroups: {}'.format(tm_end-tm_start))
        try:
            # dfsurl = asyncio.run(myDfs.savetoDfs(img_string))
            tm_start=time.time()
            dfsurl = loop.run_until_complete(groups)
            tm_end = time.time()
            print ('name: {}, tmall: {}, rst: {}'.format(img_name, tm_end - tm_start, dfsurl))
        except CustomError as e:
            print ('Excep: {}'.format(e))
    loop.close()

