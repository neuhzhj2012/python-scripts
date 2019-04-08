#encoding: utf-8
import datetime
import time
import MySQLdb

class SQLOP:
    def __init__(self):
        # 打开数据库连接
        host = 'localhost'
        user = 'root'
        passport=''
        database='design_ai'
        self.db = MySQLdb.connect(host, user, passport, database, charset='utf8',autocommit=True)
        self.cursor = self.db.cursor() # 使用cursor()方法获取操作游标
    def __del__(self):
        # 关闭数据库连接
        self.db.close()
    def __execute(self, sqlSequence, value=None):
        if None == value:
            self.cursor.execute(sqlSequence)
        else:
            if isinstance(value, list):
                self.cursor.executemany(sqlSequence, value) #同时插入多个语句
            else:
                self.cursor.execute(sqlSequence, value)  #插入一条语句

    def getVersion(self):
        self.__execute("SELECT VERSION()")
        data = self.cursor.fetchone() # 使用 fetchone() 方法获取一条数据
        # print "Database version : %s " % data
        return data
    def getTables(self): #获得当前数据库中的表
        self.cursor.execute('show tables')
        table_list = [tuple[0] for tuple in self.cursor.fetchall()] # 使用 fetchall() 方法获取所有结果
        return table_list

    def getTableStatus(self, tableName):
        stmt = "SHOW TABLES LIKE '%s'"%tableName
        self.__execute(stmt)
        result = self.cursor.fetchone()
        # print ('%s exists: %s'%(tableName, result)) #不存在时为None, 存在时为(u'$tableName')
        return result
    def dropTable(self, tableName):
        self.__execute("DROP TABLE IF EXISTS %s" % (tableName))
    def createTable(self, sqlSequence, tableName):
        '''
        :param sqlSequence: 创建数据表的sql语句
        :param tableName:  表名
        :return:
        '''
        # 由于python区分大小写，所以self.getTables()方法无法判断表是否存在
        #assert tableName not in self.getTables(),'%s exists '%(tableName)
        assert self.getTableStatus(tableName)==None, '%s exists '%(tableName)
        self.dropTable(tableName) #删除数据表
        self.__execute(sqlSequence)
    def insertTable(self, sqlSequence, value):
        self.__execute(sqlSequence, value)

if __name__=='__main__':
    mySqlObject = SQLOP()

    #增 创建数据表SQL语句
    mytable='EMPLOYEE'
    sql = """CREATE TABLE EMPLOYEE (
             FIRST_NAME  CHAR(20) NOT NULL,
             LAST_NAME  CHAR(20),
             AGE INT,
             SEX CHAR(1),
             INCOME FLOAT )"""
    # mySqlObject.createTable(sql, "EMPLOYEE")

    #删 对应数据表
    for idx in range(2):
        tableName = 'EMPLOYEE'+str(idx+2)
        mySqlObject.dropTable(tableName)

    #改 插入新数据
    # mytable = 'jiaodiantu2'
    # caller = '1'
    # device = '1'
    # dst_img = 'https://img3.autoimg.cn/g30/M08/51/C5/ChsEoFw1fRqAAGnvAAQ9_Zr3hic328.jpg'
    # status = 'True'
    # Created_STime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #
    # sql = """INSERT INTO jiaodiantu2 (caller, \
    #          device, dst_img, status, Created_STime) \
    #          VALUES (%s, %s, %s, %s, %s)"""
    # sql = " INSERT INTO " + mytable + "(caller, \
    #          device, dst_img, status, Created_STime) \
    #          VALUES (%s, %s, %s, %s, %s)"
    # value = (int(caller), int(device), dst_img, status, Created_STime)


    # sql = """INSERT INTO employee(FIRST_NAME, \
    #        LAST_NAME, AGE, SEX, INCOME) \
    #        VALUES (%s, %s, %s, %s, %s )"""
    sql = """INSERT INTO employee(FIRST_NAME, \
           LAST_NAME, AGE, SEX, INCOME) \
           VALUES ('Macle', 'Mohan', 19, 'F', 2000)"""
    # value = ('Macle', 'Mohan', 19, 'F', 2000)
    # values =  [('Mac', 'Mohan', 20, 'M', 2000),
    #           ('Macle', 'Mohan', 19, 'F', 2000),
    #           ('Jodan', 'Mohan', 20, 'M', 1000)]
    value=None
    mySqlObject.insertTable(sql, value)

    #查 查看所有数据表
    print 'db version: ', mySqlObject.getVersion() #数据库版本信息
    print 'tables: ', mySqlObject.getTables() #显示所有数据表
    print '%s exists: %s'%(mytable, mySqlObject.getTableStatus(mytable)) #查看当前数据表是否存在






