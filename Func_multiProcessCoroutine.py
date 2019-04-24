# encoding: utf-8
import requests
from multiprocessing import Process
import gevent
import sys
import os
import numpy
import urllib
import time
from gevent import monkey

monkey.patch_all()

def fetch(url, name, folder):
    abspath=os.path.join(folder, name)
    try:
        tm_start=time.time()
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
        print 'name: {} done, tm_svc: {}'.format(name, tm_end-tm_start)
    except Exception, e:
        print "####name: {}, error: {}####".format(name,e)

        if os.path.exists(abspath):
            os.remove(abspath)

def process_start(url_list, folder):
    tasks = []
    for idx, urlinfo in enumerate(url_list):
        name, url=urlinfo.split()
        tasks.append(gevent.spawn(fetch, url,name, folder))
    gevent.joinall(tasks)  # 使用协程来执行


def task_start(filepaths, batch_size=5, folder='./tmp'):  # 每batch_size条filepaths启动一个进程
    num=len(filepaths)

    if not os.path.exists(folder):
        os.makedirs(folder)

    for idx in range(num / batch_size):
        url_list = filepaths[idx * batch_size:(idx + 1) * batch_size]
        p = Process(target=process_start, args=(url_list, folder,))
        p.start()

    if num % batch_size > 0:
        idx = num / batch_size
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


