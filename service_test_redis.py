#encoding: utf-8
'''
python service_test_redis.py -a check -v test_online
python service_test_redis.py -a add -v test_tmp -k 123
python service_test_redis.py -a delete -v test_tmp
python service_test_redis.py -a add -v test_tmp
'''
import os, sys, cv2
import base64, io
import requests
import argparse
import sys
import uuid
import numpy as np
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

if sys.version_info[0] == 3:
    import simplejson as json
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
else:
    import json

def getArgs():
    args = argparse.ArgumentParser()
    args.add_argument('-a', '--action', help='add or delete or check',type=str,
                      default='check')
    args.add_argument('-v', '--value', help='service type',type=str, default=None)
    args.add_argument('-k', '--key', help='service key',type=str, default=None)

    return vars(args.parse_args())

if __name__ == '__main__':
    args = getArgs()
    ops = args['action']
    service_type = args['value']
    service_key = args['key']

    flag_redis = True
    host_health = 'http://127.0.0.1:8001/health'
    host_redis = 'http://127.0.0.1:8001/redis'

    if flag_redis:
        host = host_redis
    else:
        host = host_health

    data = dict()
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    #warmup
    for i in range(1):
        svc_rtn = session.get(host)

        print('id: {}, info: {}'.format(i, svc_rtn.text))

    key = 'zhimakaimen'
    if sys.version_info[0] == 3:
        headers={'Authorization':'Basic' + base64.b64encode(key.encode()).decode()}
    else:
        headers = {'Authorization': 'Basic' + base64.b64encode(key)}

    tm_sum = 0

    data['user']= 'test_online'
    data['password']= 'c793b8a1-c21c-11e8-8084-68f728d8fd1e'
    data['user']= 'dmcv'
    data['password']= '06b24fa1-3c70-11e8-9281-acde48001122'
    data['ops']= ops #add or delete
    data['service_type'] = service_type
    data['service_key'] = service_key


    svc_rtn = requests.post(host, headers=headers, data = data)
    tm_svc = svc_rtn.elapsed.total_seconds()
    tm_sum += tm_svc
    svc_rtn = eval(svc_rtn.text)
    print ('rst: {}'.format(svc_rtn))


