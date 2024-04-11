import json
import pandas as pd
import requests
import pymysql
import numpy as np
from base.mysql_conn import get_conn

'''
headers = {'content-type': 'application/json',"Accept": "application/json"}
raw = requests.get(url="http://192.168.2.186:9980/Schedule/getPoint",headers=headers)
data = raw.json()
'''


def read_data_from_mysql_for_di_pressure():
    # 连接数据库
    conn = get_conn()

    # 从数据库读取所有站点数据
    def get_data():
        sql = """
            SELECT a.tagv_name, a.tagv_value, a.tagv_fresh_time FROM work.fz_tag_view a
            INNER JOIN work.fz_tag_station b 
            on a.tagv_name =b.tag_name 
            WHERE a.tagv_status = '0' 
            AND a.tagv_value >= 0 
            AND b.type = 1
            AND a.tagv_name in ('B_RYX_RZZ_FT1111','B_RYX_RZZ_PT1111',
            'B_RYX_DHZ_PT2101','B_RYX_DHZ_PT2111',
            'B_RYX_HAZ_PT3101','B_RYX_HAZ_PT3111',
            'B_RYX_GYZ_PT4101','B_RYX_GYZ_PT4101',
            'B_RYX_GYZ_PT4111','B_RYX_YZZ_PT5101')
            ORDER BY a.tagv_fresh_time DESC 
            LIMIT 0,5000
        """  # 取站点列表中前8000条非0值
        with conn.cursor() as cursor:
            cursor.execute(sql)
            sql_data = cursor.fetchall()
            # print(result)
        return sql_data

    sql_data = get_data()
    # 对数据按照站点进行分列
    rz_of_value = []  # 日照出站流量
    rz_of_time = []  # 日照出站流量
    rz_op_value = []  # 日照出站压力
    rz_op_time = []  # 日照出站压力
    dh_ip_value = []  # 东海进站压力
    dh_ip_time = []  # 东海进站压力
    dh_op_value = []  # 东海出站压力
    dh_op_time = []  # 东海出站压力
    ha_ip_value = []  # 淮安进站压力
    ha_ip_time = []  # 淮安进站压力
    ha_op_value = []  # 淮安出站压力
    ha_op_time = []  # 淮安出站压力
    gy_ip_value = []  # 观音进站压力
    gy_ip_time = []  # 观音进站压力
    gy_op_value = []  # 观音出站压力
    gy_op_time = []  # 观音出站压力
    yz_ip_value = []  # 仪征进站压力
    yz_ip_time = []  # 仪征进站压力
    for elem in sql_data:
        value = float(elem['tagv_value'])
        # 只取到分钟级 例如 2022-06-01 18:06
        t = elem['tagv_fresh_time'].strftime('%Y-%m-%d %H:%M')
        if elem['tagv_name'] == 'B_RYX_RZZ_FT1111':  # 日照出站流量
            if t not in rz_of_time:
                rz_of_time.append(t)
                rz_of_value.append(value)
        elif elem['tagv_name'] == 'B_RYX_RZZ_PT1111':  # 日照出站压力
            if t not in rz_op_time:
                rz_op_time.append(t)
                rz_op_value.append(value)
        elif elem['tagv_name'] == 'B_RYX_DHZ_PT2101':  # 东海进站压力
            if t not in dh_ip_time:
                dh_ip_time.append(t)
                dh_ip_value.append(value)
        elif elem['tagv_name'] == 'B_RYX_DHZ_PT2111':  # 东海出站压力
            if t not in dh_op_time:
                dh_op_time.append(t)
                dh_op_value.append(value)
        elif elem['tagv_name'] == 'B_RYX_HAZ_PT3101':  # 淮安进站压力
            if t not in ha_ip_time:
                ha_ip_time.append(t)
                ha_ip_value.append(value)
        elif elem['tagv_name'] == 'B_RYX_HAZ_PT3111':  # 淮安出站压力
            if t not in ha_op_time:
                ha_op_time.append(t)
                ha_op_value.append(value)
        elif elem['tagv_name'] == 'B_RYX_GYZ_PT4101':  # 观音进站压力
            if t not in gy_ip_time:
                gy_ip_time.append(t)
                gy_ip_value.append(value)
        elif elem['tagv_name'] == 'B_RYX_GYZ_PT4111':  # 观音出站压力
            if t not in gy_op_time:
                gy_op_time.append(t)
                gy_op_value.append(value)
        elif elem['tagv_name'] == 'B_RYX_YZZ_PT5101':  # 仪征进站压力
            if t not in yz_ip_time:
                yz_ip_time.append(t)
                yz_ip_value.append(value)
    # df = pd.DataFrame.from_dict({'tagv_name':name,'tagv_desc':desc,'tagv_value':value})
    rz_of_data = pd.DataFrame.from_dict({'time': rz_of_time, 'rzof': rz_of_value})
    rz_op_data = pd.DataFrame.from_dict({'time': rz_op_time, 'rzop': rz_op_value})
    dh_ip_data = pd.DataFrame.from_dict({'time': dh_ip_time, 'dhip': dh_ip_value})
    dh_op_data = pd.DataFrame.from_dict({'time': dh_op_time, 'dhop': dh_op_value})
    ha_ip_data = pd.DataFrame.from_dict({'time': ha_ip_time, 'haip': ha_ip_value})
    ha_op_data = pd.DataFrame.from_dict({'time': ha_op_time, 'haop': ha_op_value})
    gy_ip_data = pd.DataFrame.from_dict({'time': gy_ip_time, 'gyip': gy_ip_value})
    gy_op_data = pd.DataFrame.from_dict({'time': gy_op_time, 'gyop': gy_op_value})
    yz_ip_data = pd.DataFrame.from_dict({'time': yz_ip_time, 'yzip': yz_ip_value})
    # print(data)

    # 时间对齐
    rz_dh_time = rz_op_data.merge(dh_ip_data, on='time', how='inner')  # 日照-东海
    rz_dh_time = rz_of_data.merge(rz_dh_time, on='time', how='inner')
    rz_dh_time = drop_flow_outlier(rz_dh_time)
    #rz_dh_time.to_csv("rz_dh_time.csv")
    #print_json(rz_dh_time)
    rz_dh_time = np.array(rz_dh_time)

    #print(rz_dh_time.shape)
    dh_ha_time = dh_op_data.merge(ha_ip_data, on='time', how='inner')  # 东海-淮安
    dh_ha_time = rz_of_data.merge(dh_ha_time, on='time', how='inner')
    dh_ha_time = drop_flow_outlier(dh_ha_time)
    dh_ha_time = np.array(dh_ha_time)
    #print(dh_ha_time.shape)
    ha_gy_time = ha_op_data.merge(gy_ip_data, on='time', how='inner')  # 淮安-观音
    ha_gy_time = rz_of_data.merge(ha_gy_time, on='time', how='inner')
    ha_gy_time = drop_flow_outlier(ha_gy_time)
    ha_gy_time = np.array(ha_gy_time)
    #print(ha_gy_time.shape)
    gy_yz_time = gy_op_data.merge(yz_ip_data, on='time', how='inner')  # 观音-仪征
    gy_yz_time = rz_of_data.merge(gy_yz_time, on='time', how='inner')
    gy_yz_time = drop_flow_outlier(gy_yz_time)
    gy_yz_time = np.array(gy_yz_time)
    #print(gy_yz_time.shape)
    conn.close()  # 关闭数据库连接
    return rz_dh_time, dh_ha_time, ha_gy_time, gy_yz_time


def print_json(rz_dh_time):
    rz_dh_output = rz_dh_time
    rz_dh_output['压差'] = rz_dh_output['rzop'] - rz_dh_output['dhip']
    rz_dh_output = rz_dh_output.rename(columns={'rzof': '流量', 'rzop': '入口压力'})
    rz_dh_output = rz_dh_output[['流量', '压差', '入口压力']]
    print(rz_dh_output.to_json(index=False, force_ascii=False, orient='split'))


def drop_flow_outlier(df,column="rzof"):
    # print(df.describe())
    quantile_75 = df[column].quantile(0.75)
    quantile_25 = df[column].quantile(0.25)
    flow = df[column]
    flow[(flow >= (quantile_75 - quantile_25) * 2.5 + quantile_75) | (
                flow <= quantile_25 - (quantile_75 - quantile_25) * 2.5)] = np.nan
    df = df.dropna()
    #print(df.describe())
    return df


if __name__ == "__main__":

    read_data_from_mysql_for_di_pressure()

