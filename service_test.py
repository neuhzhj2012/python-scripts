import os, sys
import base64
import requests
import argparse
import sys
import uuid
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

if sys.version_info[0] == 3:
    import simplejson as json
else:
    import json


def getArgs():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', help='input url',
                      default='https://gss1.bdstatic.com/9vo3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike92%2C5%2C5%2C92%2C30/sign=aaa4ffe658afa40f28cbc68fca0d682a/023b5bb5c9ea15ceef87a8ddb4003af33b87b2fd.jpg')
    return vars(args.parse_args())


def getAllUrls(txt):
    rtn = list()
    with open(txt, 'r') as fp:
        buffs = fp.readlines()
    rtn = [tmp.strip() for tmp in buffs]
    return rtn

if __name__ == '__main__':
    args = getArgs()
    img_path = args['input']

    if os.path.isfile(img_path):
        img_paths = getAllUrls(img_path)  # url 文件列表
    else:
        img_paths = [img_path]

    host = 'http://localhost:7001/test/dmcvrecog'
    host_health = 'http://localhost:7001/test/health'

    data = dict()
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    for i in range(1):
        svc_rtn = session.get(host)

        print('id: {}, info: {}'.format(i, svc_rtn.text))

    key = 'zhimakaimen'
    if sys.version_info[0] == 3:
        headers={'Authorization':'Basic' + base64.b64encode(key.encode()).decode()}
    else:
        headers = {'Authorization': 'Basic' + base64.b64encode(key)}

    for img_path in img_paths:
        img_name = os.path.basename(img_path)

        data['img_url']= img_path
        data['pvid']= str(uuid.uuid4())

        svc_rtn = requests.post(host, headers=headers, json = data)
        tm = svc_rtn.elapsed.total_seconds()
        svc_rtn = eval(svc_rtn.text)
        print ('{} tm: {}, id: {}, rst: {}'.format(img_name, tm, data['pvid'], svc_rtn))