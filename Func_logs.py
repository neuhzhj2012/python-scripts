#encoding: utf-8
import os,re
import time
import logging
# from logging.handlers import TimedRotatingFileHandler
from myhandlers import TimedRotatingFileHandler

loginfo = logging.getLogger("info")
strtime = time.strftime("%Y%m%d%H", time.localtime())
formatter = logging.Formatter('%(asctime)s\tthread-%(thread)d\t%(levelname)s\t%(message)s\t', "%Y-%m-%d %H:%M:%S") #默认时间格式为毫秒级
fileTimeHandlerinfo = TimedRotatingFileHandler(filename=os.path.join('logs', 'dmcv_recog_info.' + strtime), backupCount=2, encoding='utf-8',  when="H", utc=False)
fileTimeHandlerinfo.setFormatter(formatter)
logging.basicConfig(level=logging.INFO)
loginfo.addHandler(fileTimeHandlerinfo)

#time.sleep(10)
for _ in range(5):
    for _ in range(60):
        loginfo.info("time: {}".format(time.strftime("%Y%m%d_%H%M%S", time.localtime())))
        time.sleep(60)
    print ('##epoch##')
    time.sleep(30)
