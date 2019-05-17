#encoding: utf-8
class CBASE1(object):
    def __init__(self, srcdir='./data', **kwargs):
        print srcdir
        super(CBASE1, self).__init__(**kwargs)
    def myprint(self):
        print 'cbase1'

class CBASE2(object):
    def __init__(self, size='20', **kwargs):
        print size
        super(CBASE2, self).__init__(**kwargs)
    def myprint(self):
        print 'cbase2'

class C3(CBASE1, CBASE2):
# class C3(CBASE2, CBASE1):
    def __init__(self, name='saturn', **kwargs):
        print name
        # #初始化基类函数方法1：显示调用基类函数，此时第二个基类函数被初始化两次
        # CBASE1.__init__(self,srcdir=kwargs['srcdir']) #基类继承关系为CBASE1, CBASE2时，初始化CBASE2共被初始化两次
        # CBASE2.__init__(self,size=kwargs['size'])

        #初始化基类函数方法2：使用super函数,每个类都被初始化一次，建议方式
        super(C3, self).__init__(**kwargs) #按照类继承关系逐步初始化基类
    def myprint(self):  #函数重写
        print 'cbase3'

if __name__=='__main__':
    obj = C3(name='zhzh', size='50', srcdir='./tmp')
    print '类继承关系： {}'.format(C3.mro())
    obj.myprint()