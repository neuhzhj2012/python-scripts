#encoding: utf-8

class OP_LIST():
    def __init__(self):
        pass
    def get_uniq_val(self, vals):
        return list(set(vals))

class OP_DICT():
    def __init__(self):
        pass
    def get_sort_by_key(self, vals):
        return sorted(vals.items(), key=lambda k: k[0])
    def get_sort_by_val(self, vals):
        return sorted(vals.items(), key=lambda k: k[1])


if __name__=='__main__':
    dictObj = OP_DICT()
    listObj = OP_LIST()

    tmp = {1: 3, 3: 2, 2: 1}
    print dictObj.get_sort_by_val(tmp)