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
# keywords = "骑行 摩托车, 摩托车 旅游, 摩托车 公路, 摩托车 运动, \
#            乐趣 摩托车,摩托车 北极光,摩托车 宗申,摩托车 重型机车,重型机车,\
#            重型机车 运动"

keywords = "自行车, 山地自行车,山地车, 骑行, 骑山地自行车,自行车 旅游, 自行车 运动, \
           山地车 乐趣, 自行车 乐趣"



arguments = {"keywords":keywords,"limit":5000,"print_urls":True,
             "format":"jpg", "size":">400*300","no_download":True,
             "chromedriver":"C:\\chromedriver_win32\\chromedriver.exe"}
print (arguments)
rst_jpg = response.download(arguments)
arguments = {"keywords":keywords,"limit":5000,"print_urls":True,
             "format":"png", "size":">400*300","no_download":True,
             "chromedriver":"C:\\chromedriver_win32\\chromedriver.exe"}
rst_png = response.download(arguments)

# rst = {keywords:rst_jpg[0][keywords] + rst_png[0][keywords] }

urls = []
for k,v in rst_jpg[0].items():
    urls +=v
for k,v in rst_png[0].items():
    urls +=v

print ('num: {}'.format(len(urls)))
# rst = {keywords:rst_jpg[0][keywords] + rst_png[0][keywords] }
rst = {keywords:urls }


# print ("num: {}, val: {}".format(len(rst[0]['风景']), rst[0]['风景']))

# np.save('keywords.npy', rst[0]) #保存为
np.save('bike.npy', rst) #保存为

# kv=np.load('keywords.npy').item() #数据加载