# encoding: utf-8
from multiprocessing import Process
import gevent
import sys
import os
import numpy
from gevent import monkey #调制至request前，因为有时会报RecursionError

monkey.patch_all()
import requests
import urllib.request as urllib
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time


session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

def fetch(url, subfolder, folder):
    name=os.path.basename(url)
    if len(name.split('.')) ==1:
        name +='.jpg'
    abspath=os.path.join(folder, subfolder + '_' + name)
    #if not os.path.exists(os.path.dirname(abspath)):
        #os.makedirs(os.path.dirname(abspath))
    
    if True:
    #try:
        tm_start=time.time()
        # urllib.urlretrieve(url, os.path.join(folder, os.path.basename(url)))
        #urllib.urlretrieve(url, abspath) #易下载不全

        
        #way2
        #res = requests.get(url, timeout=15)
        #res = session.get(url,headers={'User-Agent': 'firefox'}, timeout=15)
        res = session.get(url, timeout=15)
        print (int(res.status_code), url)
        assert int(res.status_code) == 200 #查看urls是否有空格

        with open(abspath.replace('png', 'jpg'), 'wb') as fp:
            fp.write(res.content)
        
        tm_end = time.time()
        print ('name: {} done, tm_svc: {}'.format(name, tm_end-tm_start))
    #except Exception as e:
    else:
        print ("####name: {}, error: {}####".format(name,e))

        if os.path.exists(abspath):
            os.remove(abspath)

def process_start(url_list, folder):
    tasks = []
    for idx, urlinfo in enumerate(url_list):
        
        url=urlinfo.decode('utf-8').split('\t')[0]
        #print (urlinfo.decode('utf-8').split('\t'))
        subfolder=urlinfo.decode('utf-8').split('\t')[1].split('_')[0]
        tasks.append(gevent.spawn(fetch, url,subfolder, folder))
    gevent.joinall(tasks)  # 使用协程来执行


def task_start(filepaths, batch_size=5, folder='./tmp'):  # 每batch_size条filepaths启动一个进程
    num=len(filepaths)

    if not os.path.exists(folder):
        os.makedirs(folder)

    for idx in range(num // batch_size):
        url_list = filepaths[idx * batch_size:(idx + 1) * batch_size]
        p = Process(target=process_start, args=(url_list, folder,))
        p.start()

    if num % batch_size > 0:
        idx = num // batch_size
        url_list = filepaths[idx * batch_size:]
        p = Process(target=process_start, args=(url_list, folder,))
        p.start()

if __name__ == '__main__':
    img_list = open('merge.csv', 'rb').readlines()
    imgs = [img.strip() for img in img_list]
    
    rstdir='carProd'
    tm_start = time.time()
    img_step = 1000
    task_start(imgs, img_step, rstdir)
    exit()
    #分阶段下载至不同文件夹
    for idx in range(0, len(imgs),img_step):
        urls = imgs[idx].split()[0]
        angle = imgs[idx].split()[1]
        urls_hdfs=[img.split()[0] for img in imgs[idx:idx+img_step]]

        rootdir = os.path.join(rstdir, angle)
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        
        task_start(urls_hdfs, 1000, rootdir)
        break
    tm_end = time.time()    
    print ('tm_task: {}'.format(tm_end - tm_start))


