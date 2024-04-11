import psycopg2
import pymysql
from base.config import mysql_host,mysql_user,mysql_db,mysql_password,mysql_port

def get_conn():
    conn = pymysql.connect(host=mysql_host, user=mysql_user,
                           password=mysql_password, db=mysql_db, port=mysql_port,
                           charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor, autocommit=True)

    return conn

# def get_conn_1():
#     conn = psycopg2.connect(
#         dbname=mysql_db,
#         user=mysql_user,
#         password=mysql_password,
#         host=mysql_host,
#         port=mysql_port)
#     return conn

get_conn()