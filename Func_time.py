import time


print ('{} second to January 1st, 1970'.format(int(time.time()))) #距离计时起点开始多少秒
print ('int to time: {}'.format(time.gmtime(1))) #秒转换为计时时间
print ('time format output: {}'.format(time.strftime('%H:%M:%S:%s', time.gmtime(1)))) #格式化输出时间
print ('format output: {}'.format(time.strptime('12:21:11','%H:%M:%S')))
print (time.strftime("%Y%m%d_%H%M%S",time.gmtime(1567146428)))  #混合输出