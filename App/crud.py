import pymysql
import os
from dotenv import load_dotenv

load_dotenv()
host = os.getenv('HOST')
user = os.getenv('USER')
password = os.getenv('PASSWORD')
db = os.getenv('DB')

def connect(sql):
    con=pymysql.connect(host=host, user=user, password=password ,db=db)
    a=con.cursor()
    a.execute(sql)
    return con,a

def select():
    jmlh=[0]
    nama=[0]
    con,a=connect("Select user_id,product_id,rating from ratings")
    con=a.fetchall()
    return con

if __name__ == '__main__':
  a = select()
  print(len(a))