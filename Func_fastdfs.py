#encoding: utf-8
import time
import requests
from Func_utils import *
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class AUTOHOMEDFS(object):
    fastdfs_url = 'http://baseup.com.cn/service/fastUp'  #上传图片地址
    fastdfs_url_get_token = 'http://baseup.com.cn/service/get/token' #token获取地址
    fastdfs_key = '305c300d06099de7ab92693369af9a6762c655dae5cdd4061c01'
    fastdfs_source = 'dmcv'
    tm_token_valid = 1 #有效单位：分钟，默认为24h

    # url_prefix = 'https://img3.autoimg.cn/'
    url_prefix = 'http://img3.img.cn/dwdfs/'


    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    def __init__(self, when='h', interval=1):   #由于token最长有效时间为24小时，且单位为minutes,故when(时间单位)只有h, m,自定义个MIDNIGHT(半夜,需要配合tm_token_valid为24*60小时)
        super().__init__()

        self.when = when.upper()
        if self.when=='S':
            self.interval = 1
        elif self.when == 'M':
            self.interval = 60 # one minute
        elif self.when == 'H':
            self.interval = 60 * 60 # one hour
        else:
            self.interval = 24 * 60 * 60  # one hour

        self.interval = self.interval * interval
        if self.when != 'S':
            self.tm_token_valid *= (self.interval // 60) #tm_token_valid类型为整数

        self.payload = {'source': self.fastdfs_source, 'minutes': self.tm_token_valid}
        self.payload['key'] = self.fastdfs_key
        self.token = None
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
        result = self.session.post(self.fastdfs_url_get_token, data=self.payload,timeout=1) #1s超时
        # print ('fastdfs tm_post: {}'.format(result.elapsed.total_seconds()))
        result = eval(result.text)
        assert result['code']==0,'get fastdfs token error'
        # print (result)
        self.token = result['result']

    def savetoDfs(self, img_string, extname='jpg'):
        # tm_start = time.time()
        # if self.shouldRollover(): #实际使用中，按时间获取token时多次导致阻塞
        #     # print ('doRollover')
        #     self.reset()
        self.getToken()
        # tm_end = time.time()
        # print ('tm shouldRollover: {}'.format(tm_end - tm_start))
        payload = {'source': self.fastdfs_source, 'exName': extname, 'token': self.token}
        # result = requests.post(self.fastdfs_url, files={'file': img_string}, data=payload)
        result = self.session.post(self.fastdfs_url, files={'file': img_string}, data=payload,timeout=1)
        # print ('tm_dfs_post: {}, rst: {}'.format(result.elapsed.total_seconds(), eval(result.text)))
        result = eval(result.text)

        if result['code'] != 0:
            self.reset()
            raise CustomError('fastdfs error {}'.format(result)) #主动抛出异常
        return self.url_prefix + result['result']

    def downloadImg(self, img_url, timeout=1):
        if (img_url[:4] != "http"): #兼容直接
            # img_url = "http://img3.autoimg.cn/dwdfs/" + img_url;
            img_url = self.url_prefix + img_url

        result = self.session.get(img_url, timeout=timeout)
        # print('tm_dfs_get: {}'.format(result.elapsed.total_seconds()))
        return bytes2cv(result.content)


if __name__ == '__main__':
    myDfs = AUTOHOMEDFS(when='m', interval=1)
    img_path = '0.jpg'
    for _ in range(3):
        for _ in range(2):
            try:
                rst = myDfs.savetoDfs(open(img_path,'rb').read())
                print ('rst: {}'.format(rst))
            except CustomError as e:
                print ('Excep: {}'.format(e))
        print ('\n\n')
        time.sleep(60)

