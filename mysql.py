import pymysql
import numpy
import matplotlib
import sklearn
import pandas
db=pymysql.connect(host="localhost",user="root",password="123456",database="mydb"
                   ,charset="utf8")
cursor=db.cursor()
# 查询
sql="select * from shop where grade=%s"
data=8
cursor.execute(sql,data)
for i in cursor.fetchall():
    print(i)

# # 插入
sql="insert into shop(id,name,account,password,food,grade,profit) values(%s,%s,%s,%s,%s,%s,%s)"
data=[("11","啊臻米饭","212","223","啤酒鸭 45 回锅肉 28 宫保鸡丁 30 水滑肉 30","10","100"),
      ("12","辣凤芹米饭","212","223","啤酒鸭 45 回锅肉 28 宫保鸡丁 30 水滑肉 30","10","100")]
# executemany 处理多个语句 execute 处理单个语句
cursor.executemany(sql,data)
db.commit()   # 提交，不然无法保存插入或者修改的数据(这个一定不要忘记加上)

# 修改数据
# %s加不加括号主要看语句是在中间还是在最后，最后就要加
sql="update shop set grade=%s where name=%s"
data=("10","新疆味道")
cursor.execute(sql,data)
db.commit()

# 删除数据
sql="delete from shop where id=%s"
data=(12)
cursor.execute(sql,data)
db.commit()
cursor.close()  # 关闭游标
db.close()  # 关闭连接


