import pandas as pd
from base.mysql_conn import get_conn
import numpy as np



def drop_outlier(df, column="in"):
    #print(df.describe())

    quantile_75 = df[column].quantile(0.75)
    quantile_25 = df[column].quantile(0.25)
    flow = df[column]
    flow[(flow >= (quantile_75 - quantile_25) * 2.5 + quantile_75) | (flow <= quantile_25 - (quantile_75 - quantile_25) * 2.5)] = np.nan
    df = df.dropna()

    #print(df.describe())
    return df


def drop_outlier_with_width(df, column="in", width = 2.5):
    #print(df.describe())

    quantile_75 = df[column].quantile(0.75)
    quantile_25 = df[column].quantile(0.25)
    flow = df[column]
    flow[(flow >= (quantile_75 - quantile_25) * width + quantile_75) | (
                flow <= quantile_25 - (quantile_75 - quantile_25) * width)] = np.nan
    df = df.dropna()

    #print(df.describe())
    return df


def get_tag_value_diff_with_time(tag1_name, tag2_name,minutes=60, freq='1MIN'):
    last_time = get_last_time(tag1_name)
    conn = get_conn()

    sql="""
    select tagv_value,tagv_name,tagv_fresh_time from fz_tag_view t
where t.tagv_name in ('{}','{}')
and t.tagv_status = 0
and t.tagv_fresh_time > ('{}' - interval {} minute) 
    """.format(tag1_name,tag2_name, last_time, minutes)

    values = []
    time = []
    name = []

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        for elem in data:
            values.append(float(elem['tagv_value']))
            time.append(elem['tagv_fresh_time'])
            name.append(elem['tagv_name'])

    result = []
    result = pd.DataFrame(result)
    if len(name) == 0:
        return result

    df = pd.DataFrame.from_dict({'name':name,'value':values,'time':time})
    df['time'] = df['time'].dt.round(freq)

    df1 = df[df['name']==tag1_name].rename({'value':'value1'},axis=1)
    df2 = df[df['name']==tag2_name].rename({'value':'value2'},axis=1)

    result = df1.merge(df2,on='time',how='inner')
    result['diff'] = result['value2'] - result['value1']

    result = result[['time','diff']]

    conn.close()
    return result




def get_tag_value_diff_with_time_and_origin(tag1_name, tag2_name,minutes=60, freq='1MIN'):
    last_time = get_last_time(tag1_name)
    conn = get_conn()

    sql="""
    select tagv_value,tagv_name,tagv_fresh_time from fz_tag_view t
where t.tagv_name in ('{}','{}')
and t.tagv_status = 0
and t.tagv_value > 0
and t.tagv_fresh_time > ('{}' - interval {} minute) 
    """.format(tag1_name,tag2_name, last_time, minutes)

    values = []
    time = []
    name = []

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        for elem in data:
            values.append(float(elem['tagv_value']))
            time.append(elem['tagv_fresh_time'])
            name.append(elem['tagv_name'])

    df = pd.DataFrame.from_dict({'name':name,'value':values,'time':time})
    df['time'] = df['time'].dt.round(freq)

    df1 = df[df['name']==tag1_name].rename({'value':'value1'},axis=1)
    df2 = df[df['name']==tag2_name].rename({'value':'value2'},axis=1)

    result = df1.merge(df2,on='time',how='inner')
    result['diff'] = result['value2'] - result['value1']

    result = result[['time', 'value1', 'value2','diff']]

    conn.close()
    return result

def get_single_tag_value_config(tag_name):
    conn = get_conn()
    sql = """
        select tagc_fz08 from fz_tag_config
    where tagc_name = '{}' 
    """.format(tag_name)
    values = []
    with conn.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()
    df = pd.DataFrame(result)
    df.columns = ['value']
    #     for elem in result:
    #         values.append(float(elem['tagv_value']))
    # df = pd.DataFrame.from_dict({'value': values})

    conn.close()

    return df

def get_single_tag_value(tag_name, minutes=60):
    last_time = get_last_time(tag_name)
    conn = get_conn()

    sql="""
    select tagv_value from fz_tag_view t
where t.tagv_name = '{}'
and t.tagv_status = 0
and t.tagv_value > 0
and t.tagv_fresh_time > ('{}' - interval {} minute) 
    """.format(tag_name, last_time, minutes)
    print('getflow',sql)
    values = []

    with conn.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()

        for elem in result:
            values.append(float(elem['tagv_value']))

    df = pd.DataFrame.from_dict({'value':values})

    conn.close()

    return df


def get_last_time(tag_name):
    conn = get_conn()

    sql="""
    select tagv_fresh_time from fz_tag_view t
where t.tagv_name = '{}'
and t.tagv_status = 0
ORDER BY t.tagv_fresh_time
DESC LIMIT 0,1 
    """.format(tag_name)
    # print('getlasttime',sql)

    fresh_time = str('0')
    with conn.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()

        for elem in result:
            fresh_time = str(elem['tagv_fresh_time'])

    conn.close()

    return fresh_time


def get_last_tag_value(tag_name):
    conn = get_conn()

    sql="""
    select tagv_value from fz_tag_view t
where t.tagv_name = '{}'
and t.tagv_status = 0
and t.tagv_value > 0
ORDER BY t.tagv_fresh_time
DESC LIMIT 0,1 
    """.format(tag_name)

    values = []

    with conn.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()

        for elem in result:
            values.append(float(elem['tagv_value']))

    df = pd.DataFrame.from_dict({'value':values})

    conn.close()

    return df


def get_single_tag_value_with_time(tag_name, minutes=60, freq='1MIN'):
    last_time = get_last_time(tag_name)
    conn = get_conn()

    sql="""
    select tagv_value,tagv_fresh_time from fz_tag_view t
where t.tagv_name = '{}'
and t.tagv_status = 0
and t.tagv_value > 0
and t.tagv_fresh_time > ('{}' - interval {} minute) 
    """.format(tag_name, last_time, minutes)

    values = []
    time = []

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        for elem in data:
            values.append(float(elem['tagv_value']))
            time.append(elem['tagv_fresh_time'])

    df = pd.DataFrame.from_dict({'time':time,'value':values})
    if(not df.empty):
        df['time'] = df['time'].dt.round(freq)

    conn.close()

    return df




import datetime

def write_optimize_model_result_to_mysql(model_id, input_json, info_json, output_json):
    conn = get_conn()
    sql="insert into fz_model_res(`model_id`,`input`,`information`,`output`,`run_status`,`run_time`) " \
        "values(%s,%s,%s,%s,%s,%s)"

    with conn.cursor() as cursor:
        cursor.execute(sql,(model_id, input_json, info_json, output_json, "成功",datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        print("insert success")

    conn.close()


def write_failed_result_to_mysql(model_id, input_json, info_json, output_json):
    conn = get_conn()
    sql = "insert into fz_model_res(`model_id`,`input`,`information`,`output`,`run_status`,`run_time`) " \
          "values(%s,%s,%s,%s,%s,%s)"

    with conn.cursor() as cursor:
        cursor.execute(sql, (model_id, input_json, info_json, output_json, "失败",
                             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        print("insert success")

    conn.close()


def write_data_problem_result_to_mysql(model_id, input_json, info_json, output_json):
    conn = get_conn()
    sql = "insert into fz_model_res(`model_id`,`input`,`information`,`output`,`run_status`,`run_time`) " \
          "values(%s,%s,%s,%s,%s,%s)"

    with conn.cursor() as cursor:
        cursor.execute(sql, (model_id, input_json, info_json, output_json, "数据缺失",
                             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        print("insert success")

    conn.close()

def write_bump_model_result_to_mysql(beng_id, model_func, model_param, model_type, model_metrics, data):
    conn = get_conn()
    sql="insert into fz_beng_model(`beng_id`,`model_func`,`model_param`,`model_type`," \
        "`model_metrics`,`model_array`) " \
        "values(%s,%s,%s,%s,%s,%s)"

    with conn.cursor() as cursor:
        cursor.execute(sql,(beng_id, model_func, model_param, model_type, model_metrics,data))
        conn.commit()
        print("insert success")

    conn.close()







