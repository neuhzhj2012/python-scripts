#encoding: utf-8
'''
py2无法处理中文，可换成py3
'''
import sys
print ('encode: {}, path: {}'.format(sys.getdefaultencoding(), sys.path ))

from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()

# keywords = '中控,汽车, 汽车尾部 排气管,前雾灯,尾灯,仪表盘,油箱口'
keywords = "中网, 汽车"

arguments = {"keywords":keywords,"limit":110,"print_urls":True,
             "format":"jpg", "size":">400*300",
             "chromedriver":"C:\\chromedriver_win32\\chromedriver.exe"}
print (arguments)
rst = response.download(arguments)
print ("num: ",len(rst[0]['中网']))