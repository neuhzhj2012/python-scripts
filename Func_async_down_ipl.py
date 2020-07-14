#encoding: utf-8
#py3.7
import cv2, os
import time
import asyncio
import aiohttp
import requests
import numpy as np
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

'''
下载图片
'''

class DOWN():
    def __init__(self):
        self.session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.loop = asyncio.get_event_loop()
        pass
    def pull(self, imgurl):
        res = self.session.get(imgurl, timeout=1)
        img_buff = res.content

        img_np = np.asarray(bytearray(img_buff), np.uint8)
        img_tocv = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
        return [[imgurl, img_tocv]] #仅为了测试
        # return [imgurl, img_tocv]

    async def pull_async(self, session, url, timeout=1):
        async with session.get(url,  timeout=timeout) as response:
            try:
                # print('pull prob1: {}'.format(response))
                text = await response.read()
                img_np = np.asarray(bytearray(text), np.uint8)
                # img_np = np.frombuffer(text, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
            except Exception as e:
                print ('##pull: {}##'.format(str(e)))
                img = None
            return img
    async def pull_batch(self, imgurls):
        tasks = list()
        async with aiohttp.ClientSession() as session:
            for imgurl in imgurls:
                task = asyncio.create_task(self.pull_async(session, imgurl))
                tasks.append(task)

            rst = await asyncio.gather(*tasks, return_exceptions=True)
            print (type(rst[0]))
        return zip(imgurls, rst)

if __name__ == '__main__':
    downObj = DOWN()

    img_url = 'http://img4.duitang.com/uploads/item/201407/21/20140721185024_Rxnva.thumb.700_0.jpeg'
    img_urls = ['http://img4.duitang.com/uploads/item/201407/21/20140721185024_Rxnva.thumb.700_0.jpeg',
                'http://www.5257love.com/zb_users/upload/2020/01/20200108135411157846285172862.png',
                'http://img.pconline.com.cn/images/upload/upc/tx/softbbs/1011/03/c0/5733206_1288751814869_1024x1024soft.jpg',
                'http://img2.tbcdn.cn/tfscom/i2/376455440/TB2n14zaItnpuFjSZFKXXalFFXa_%21%21376455440.jpg',
                'http://pic.qqtn.com/up/2018-3/15210118368465727.jpg',
                'http://b-ssl.duitang.com/uploads/item/201802/04/20180204214937_glnfe.jpeg',
                'http://photo.tuchong.com/427085/g/11483686.jpg',
                'http://gw.alicdn.com/tfscom/tuitui/i2/T1UKDHXgprXXXqkhI8_070655.jpg',
                'http://www.1-eye.cn/img.php?img.duoziwang.com/2018/06/2018010128172497.jpg',
                'http://img10.360buyimg.com/imgzone/jfs/t2512/290/48355027/146279/944abffb/5634c0a2N73b9ef2c.jpg',
                'http://b-ssl.duitang.com/uploads/item/201510/11/20151011121007_hjsVu.jpeg']

    flag_batch = True
    tm_start = time.time()
    if not flag_batch:
        for img_url in img_urls:
            rst = downObj.pull(img_url)
    else:
        rst = asyncio.run(downObj.pull_batch(img_urls))
    tm_svc = time.time() - tm_start
    print('tm: {}'.format(tm_svc))
    for i, (img_url, img) in enumerate(rst):
        if (not isinstance(img, (np.ndarray))):
            print ('img_url: {} download error'.format(img_url))
            continue
        # print(img.shape)
        cv2.imwrite(str(i) + '_' + os.path.basename(img_url), img)
    print('num: {}, tm_ave: {}'.format(len(img_urls), tm_svc/len(img_urls)))