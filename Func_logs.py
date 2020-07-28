#encoding: utf-8
import os,re
import time
import logging
# from logging.handlers import TimedRotatingFileHandler
from myhandlers import TimedRotatingFileHandler

log_dir='/workspace/logs/'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
# loginfo = logging.getLogger("info")
loginfo = logging.getLogger("Service")
strtime = time.strftime("%Y%m%d%H", time.localtime())
formatter = logging.Formatter('%(asctime)s\tthread-%(thread)d\t%(levelname)s\t\t%(message)s', "%Y-%m-%d %H:%M:%S")
fileTimeHandlerinfo = TimedRotatingFileHandler(filename=os.path.join(log_dir, 'dmcv_main.' + strtime),backupCount=2*24, encoding='utf-8', when="H", utc=False) #保留3天的日志量
fileTimeHandlerinfo.suffix = "%Y%m%d%H"
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
