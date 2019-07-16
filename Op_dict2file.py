#encoding: utf-8
#来源：https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
import numpy as np
import pickle

class DICT2FILE():
    def __init__(self):
        pass
    def saveByNp(self, varis, name='keywords.npy'):
        '''
        文件格式：.npy
        :param varis:
        :param name:
        :return:
        '''
        print ('type_var: {}, {}'.format(type(varis), type(dict())))
        assert type(varis)==type(dict())
        np.save(name, varis)
    def loadByNp(self, name='keywords.npy'):
        return np.load(name).item()


    def saveByPkl(self, varis, name='keywords.pkl'):
        with open(name, 'wb') as fp:
            pickle.dump(varis, fp, pickle.HIGHEST_PROTOCOL)
    def loadByPkl(self, name='keywords.pkl'):
        with open(name, 'rb') as fp:
            return pickle.load(fp)



if __name__=='__main__':
    dictObj = DICT2FILE()
    test1={'1':'亚洲', '2':'欧洲'}
    dictObj.saveByNp(test1, 'key.npy')
    print (dictObj.loadByNp('key.npy'))

    dictObj.saveByPkl(test1, 'key.pkl')
    print (dictObj.loadByPkl('key.pkl'))