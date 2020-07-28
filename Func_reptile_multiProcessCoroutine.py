## coding=utf-8
"""py2"""
import re
import os
import sys
import time
import numpy
import urllib
import requests
import gevent
from gevent import monkey
from multiprocessing import Process
monkey.patch_all()

class DOWN(object):
    def __init__(self):
        super(DOWN, self).__init__()

    def fetch(self, url, name, folder):
        abspath = os.path.join(folder, name)
        try:
            tm_start = time.time()
            # urllib.urlretrieve(url, os.path.join(folder, os.path.basename(url)))
            urllib.urlretrieve(url, abspath)

            '''
            #way2
            res = requests.get(imgUrl, timeout=15)
            assert int(res.status_code) == 200

            with open(abspath, 'wb') as fp:
                fp.write(res.content)
            '''
            tm_end = time.time()
            print('name: {} done, tm_svc: {}'.format(name, tm_end - tm_start))
        except Exception as e:
            print("####name: {}, error: {}####".format(name, e))

            if os.path.exists(abspath):
                os.remove(abspath)

    def processStart(self, url_list, folder):
        tasks = []
        for idx, urlinfo in enumerate(url_list):
            url, name = urlinfo, os.path.basename(urlinfo)
            tasks.append(gevent.spawn(self.fetch, url, name, folder))
        gevent.joinall(tasks)  # 使用协程来执行

    def taskStart(self, filepaths, batch_size=5, folder='./tmp'):  # 每batch_size条filepaths启动一个进程
        num = len(filepaths)

        if not os.path.exists(folder):
            os.makedirs(folder)

        for idx in range(num / batch_size):
            url_list = filepaths[idx * batch_size:(idx + 1) * batch_size]
            p = Process(target=self.processStart, args=(url_list, folder,))
            p.start()

        if num % batch_size > 0:
            idx = num / batch_size
            url_list = filepaths[idx * batch_size:]
            p = Process(target=self.processStart, args=(url_list, folder,))
            p.start()

class MYIMG(DOWN):
    def __init__(self):
        super(MYIMG, self).__init__()
        self.host_baidu = r'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1497491098685_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1497491098685%5E00_1519X735&word='

    def getUrlsByNext(self, onepageurl):
        """获取单个翻页的所有图片的urls+当前翻页的下一翻页的url"""
        if not onepageurl:
            print('已到最后一页, 结束')
            return [], ''
        try:
            html = requests.get(onepageurl).text  # unicode需要转换为utf-8编码格式
            html = html.encode('utf-8')
        except Exception as e:
            print(e)
            pic_urls = []
            fanye_url = ''
            return pic_urls, fanye_url
        pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
        # fanye_urls = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'), html, flags=0)
        pattern = '<a href="(.*)" class="n">下一页</a>'
        fanye_urls = re.findall(re.compile(pattern), html)
        fanye_url = 'http://image.baidu.com' + fanye_urls[0] if fanye_urls else ''
        return pic_urls, fanye_url

    def getAllUrlsFromBaidu(self, keywords):
        all_pic_urls = []
        default_urls = 800

        for keyword in keywords:
            fanye_count = 1  # 累计翻页数
            url_init = self.host_baidu + urllib.quote(keyword, safe='/')
            onepage_urls, fanye_url = self.getUrlsByNext(url_init)
            if fanye_url == '' and onepage_urls == []:
                continue
            print('%s 第%s页' % (keyword, fanye_count))
            all_pic_urls.extend(onepage_urls)
            while 1:
                onepage_urls, fanye_url = self.getUrlsByNext(fanye_url)
                fanye_count += 1

                if fanye_url == '' and onepage_urls == []:
                    break
                print('%s 第%s页' % (keyword, fanye_count))
                all_pic_urls.extend(onepage_urls)
                if fanye_count > default_urls:
                    break
        return list(set(all_pic_urls))

    def downImgs(self, urls, folder='imgs'):
        self.taskStart(urls, batch_size=len(urls)/6, folder=folder)


if __name__ == '__main__':
    reptileObj = MYIMG()

    keywords = ['自行车','自行车 自然风景','自行车 人文风景', '自行车 历史遗迹']  # 关键词, 改为你想输入的词即可, 相当于在百度图片里搜索一样

    imgs_baidu = "baidu"
    urlsBaidu = reptileObj.getAllUrlsFromBaidu(keywords)
    reptileObj.downImgs(urlsBaidu, imgs_baidu)

