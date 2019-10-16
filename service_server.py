# encoding=utf-8
import os, time, logging, argparse, sys
import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado import gen
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from myhandlers import TimedRotatingFileHandler

if int(sys.version[0]) == 2:
    import json
else:
    import simplejson as json

class MainHeartHandler(tornado.web.RequestHandler):
    '''
    健康检查
    '''
    def health(self):
        result = {'returncode': 0, 'message': 'OK', 'result': 'Health Check Access!'}
        self.write(json.dumps(result))
        self.finish()

    def get(self):
        self.health()

    def post(self):
        self.health()

class TornadoHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(4)

    # @tornado.web.asynchronous
    @gen.coroutine  #自动调用 self.finish() 结束请求,可以不用tornado.web.asynchronous装饰器
    def get(self):
        params = self.request.body_arguments
        print ('keys: {}, type: {}, key: {}'.format(params.keys(),(params['service_type'][0]).decode('utf-8'), params['service_key'][0].decode('utf-8')))
        print ('type equal: {}'.format(params['service_type'][0].decode('utf-8')=='tmp'))
        try:
            class_sync = [{'id':0, 'name':'unknown'}, {'id':1, 'name':'known'}]
            result = {'message': 'OK', 'returncode': 0, 'result': class_sync}
        except:
            result = {'message': 'request body error', 'returncode': 10003, 'result': 0}

        self.write(json.dumps(result))
        # self.finish()
        return

    @gen.coroutine
    # @run_on_executor
    def post(self):
        yield self.implement()
        # res = yield self.implement()

    @run_on_executor
    def implement(self):
        args = get_args()
        try:
            params = json.loads(self.request.body)

            if 'pvid' in params:
                pvid = str(params['pvid'])
            else:
                pvid = "no_pvid"

            image_url = params["img_url"]
        except:
            logerror.exception(str(args.port) + "\tparse arguments error\t" + "request.body: " + str(self.request.body))
            result = {'message': 'request body error', 'returncode': 10001, 'result': -1}
            self.write(json.dumps(result))
            return
        try:
            result={'message': 'OK', 'returncode': 0, 'result': '{}'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))}
            for idx in range(3):
                # print(idx, end=', ')
                time.sleep(1)
            self.write(json.dumps(result))
        except:
            logerror.exception(
                str(args.port) + "\tclassifier inner error\t" + "request.body: " + str(self.request.body))
            result = {'message': 'classifier inner error', 'returncode': 10003, 'result': -1}
            self.write(json.dumps(result))
        return


def main(port):
    loginfo.info('Listen...')
    loginfo.exception('Listen...')
    loginfo.error('Listen...')
    loginfo.warn('Listen...')
    application = tornado.web.Application([(r"/test/dmcvrecog", TornadoHandler),(r"/test/health", MainHeartHandler)])
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='7001', type=str)  # tornado服务器端口
    parser.add_argument('--log_dir', default='/workspace/logs/')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    loginfo = logging.getLogger("info")
    logerror = logging.getLogger("error")
    strtime = time.strftime("%Y%m%d%H", time.localtime())
    formatter = logging.Formatter('%(asctime)s\tthread-%(thread)d\t%(levelname)s\t%(message)s\t', "%Y-%m-%d %H:%M:%S")
    fileTimeHandlerinfo = TimedRotatingFileHandler(filename=os.path.join(log_dir, 'dmcv_recog_info.' + strtime),
                                                   backupCount=7 * 24, encoding='utf-8', when="H", utc=False)
    fileTimeHandlererror = TimedRotatingFileHandler(filename=os.path.join(log_dir, 'dmcv_recog_error.' + strtime),
                                                    backupCount=7 * 24, encoding='utf-8', when="H", utc=False)
    fileTimeHandlerinfo.setFormatter(formatter)
    fileTimeHandlererror.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO)
    loginfo.addHandler(fileTimeHandlerinfo)
    logerror.addHandler(fileTimeHandlererror)

    # recog = Classifier() #服务类函数
    main(args.port)
