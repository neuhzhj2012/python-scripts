#encoding: utf-8
import redis
import time
import uuid

class MYREDIS(object):
    service_type_prefix = 'car_recog_'
    password = 'dmcv'
    timeout_connect = 10000
    timeout_wait = 1000
    maxPoolTotal = 1024

    name_Admin = 'dmcv'
    password_Admin = '06b24fa1-9281-acde48001122'

    def __init__(self, host='dmcv-online-v2.codis.yzpsg2.in.com.cn', port='19375'):
        super().__init__()
        redisPools = redis.ConnectionPool(host=host, port=port, password=self.password,
                                          max_connections=self.maxPoolTotal, decode_responses=True,
                                          socket_timeout=self.timeout_wait,
                                          socket_connect_timeout=self.timeout_connect
                                          )
        self.connection = redis.Redis(connection_pool=redisPools)

        # self.connection = redis.Redis(host=host, port=port,password=self.password,
        #                          socket_timeout=self.timeout_wait,
        #                          socket_connect_timeout=self.timeout_connect,
        #                          decode_responses=True)
    def ftTypeName(self, service_type):
        return self.service_type_prefix + service_type

    def getKey(self, service_type):
        _type = self.ftTypeName(service_type)
        return self.connection.get(_type)

    def addType(self,service_type, service_key=None, is_force = False):
        if not is_force:
            key = self.getKey(service_type)
            if key:
                return True, key
        if not service_key:
            service_key = str(uuid.uuid4())
        flag_success = self.connection.set(self.ftTypeName(service_type), service_key)
        return flag_success, service_key

    def deleteType(self, service_type):
        key = self.getKey(service_type)
        if not key:
            return True
        rtn = self.connection.delete(self.ftTypeName(service_type))
        if rtn != 1:
            return False
        return True

    def isValidAdmin(self, name, password):  #addType, deleteType需要验证是否为dmcv账号操作
        return (name == self.name_Admin) and (password == self.password_Admin)

    def isValidUser(self, service_type, service_key):
        flag_valid = True
        # tm_start = time.time()
        key = self.getKey(service_type)
        # tm_end = time.time()
        # print ('tm_redis: {}'.format(tm_end - tm_start))
        if key == None or key != service_key:
            flag_valid = False
        return flag_valid

    def test(self):
        service_type = "test"
        service_key = "c793b8a1-c21c-8084-68f728d8"

        print('valid: {}'.format(self.isValidUser(service_type, service_key)))

if __name__ == '__main__':
    myredis = MYREDIS()
    myredis.test()

    service_type = 'mainapp'
    service_key = myredis.getKey(service_type)
    print ('type: {}, key: {}'.format(service_type, service_key))
    exit(1)

    name_Admin = 'dmcv'
    password_Admin = '06b24fa1-9281-acde48001122'
    if not myredis.isValidAdmin(name_Admin, password_Admin):
        raise ValueError('invalid Admin account!.')

    test_type = 'test_online'
    test_uuid = str(uuid.uuid4())
    # 增加
    flag, key = myredis.addType(test_type, test_uuid)
    if key != test_uuid:
        print ('type: {} exists, key: {}, if want to use new uuid, set params is_force=True'.format(test_type, rtn))
    elif flag:
        print ('add ok, type: {}, key: {}'.format(test_type, key))
    else:
        print ('##add op failed##')

    #删除
    rtn = myredis.deleteType(test_type)
    if rtn:
        print ('delete type: {} successfully'.format(test_type))
    else:
        print('##delete op failed##')