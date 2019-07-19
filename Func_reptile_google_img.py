#encoding: utf-8
'''
py2无法处理中文，可换成py3
'''
import sys
import numpy as np
print ('encode: {}, path: {}'.format(sys.getdefaultencoding(), sys.path ))

from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()

# keywords = '中控,汽车, 汽车尾部 排气管,前雾灯,尾灯,仪表盘,油箱口'
keywords = "风景,旅行风景"

arguments = {"keywords":keywords,"limit":2,"print_urls":True,
             "format":"jpg", "size":">400*300","no_download":True,
             "chromedriver":"C:\\chromedriver_win32\\chromedriver.exe"}
print (arguments)
rst = response.download(arguments)
print ("num: {}, val: {}".format(len(rst[0]['风景']), rst[0]['风景']))

np.save('keywords.npy', rst[0]) #保存为

# kv=np.load('keywords.npy').item() #数据加载