import time
from pyomo.environ import *
import sys
import os
from os.path import dirname
import numpy as np
import pandas as pd
import math
from bump_choose import get_bumps_data_mysql
from base.config import *
from datetime import datetime
import matplotlib.pyplot as plt3
from base.mysql_conn import get_conn
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")  # 忽略警告
import matplotlib.pyplot as plt

# %matplotlib inline
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
from IPython.display import display

# ——————————————————————————工频泵——————————————————————————————————————
# 工频泵功率
gongpin_yg_list = ['日仪增输日照站1P-5泵有功功率', '日仪增输日照站1P-6泵有功功率'
    , '日仪增输东海站2P-1电机功率', '日仪增输东海站2P-2电机功率'
    , '日仪增输淮安站3P-5电机功率', '日仪增输淮安站3P-6电机功率'
    , '日仪增输观音站4P-1电机功率', '日仪增输观音站4P-2电机功率']
# 工频泵压力
gongpin_yl_list = ['日仪增输日照站1P-5泵入口压力检测', '日仪增输日照站1P-5泵出口压力检测',
                   '日仪增输日照站1P-6泵入口压力检测', '日仪增输日照站1P-6泵出口压力检测'
    , '日仪增输东海站2P-1泵入口压力检测', '日仪增输东海站2P-1泵出口压力检测', '日仪增输东海站2P_2泵入口压力检测',
                   '日仪增输东海站2P_2泵出口压力检测'
    , '日仪增输淮安站3P-5泵入口压力检测', '日仪增输淮安站3P-5泵出口压力检测', '日仪增输淮安站3P_6泵入口压力检测',
                   '日仪增输淮安站3P_6泵出口压力检测'
    , '日仪增输观音站4P-1泵入口压力检测', '日仪增输观音站4P-1泵出口压力检测', '日仪增输观音站4P_2泵入口压力检测',
                   '日仪增输观音站4P_2泵出口压力检测']
time_delta = 60 * 2  # 时间差设置


def do_predict():
    """
    由flow_press_fit.py得到的press系数 。press=a*flow^2 + b*flow + c
    """
    return {0: [-0.000000094675, 0.0007256603, 0.3259694955],  # 日照1
            1: [-0.000000005331, -0.0000271323, 1.9025624965],  # 日照2
            2: [-0.000000019866, 0.0000678761, 1.7679022152],  # 东海1
            3: [-0.000000108118, 0.0008077043, 0.2038966221],  # 东海2
            4: [-0.000000041413, 0.0002705021, 1.2816978385],  # 淮安1
            5: [-0.000000056491, 0.0004074756, 0.9866817504],  # 淮安2
            6: [-0.000000044731, 0.0002942832, 1.2588246840],  # 观音1
            7: [-0.000000020706, 0.0001005700, 1.6359000000]  # 观音2
            }


# !!!!!数据拟合及其不稳定，需要人工观察图像拟合公式是否能够使用！！！！！！！！！！！
for k in range(8):
    print('开始拟合%s' % gongpin_yg_list[k][4:12])
    #     sql = """
    #         select tagv_desc,tagv_value,tagv_fresh_time from  fz_tag_view
    #     where tagv_desc in ('日照站出站流量检测','{}','{}','{}') AND tagv_fresh_time LIKE '2023-0%'
    #     ORDER BY tagv_fresh_time
    #         """.format(gongpin_yg_list[k],gongpin_yl_list[k*2],gongpin_yl_list[k*2+1])
    sql = """
            select tagv_desc,tagv_value,tagv_fresh_time from  fz_tag_view
     where tagv_desc in ('日照站出站流量检测','{}','{}','{}') AND ( tagv_fresh_time LIKE '2023%' or tagv_fresh_time LIKE '2022%')
     ORDER BY tagv_fresh_time
    """.format(gongpin_yg_list[k], gongpin_yl_list[k * 2], gongpin_yl_list[k * 2 + 1])
    print(sql)
    conn = get_conn()
    result = {}

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

    source_data = pd.DataFrame(data)
    source_data['time_stamp'] = source_data['tagv_fresh_time'].astype('str').apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())

    columns = ['%s' % gongpin_yl_list[k * 2], '%s' % gongpin_yl_list[k * 2 + 1], '%s' % gongpin_yg_list[k]]
    filter_data = pd.DataFrame(
        columns=['%s' % gongpin_yl_list[k * 2], '%s' % gongpin_yl_list[k * 2 + 1], '日照站出站流量检测',
                 '%s' % gongpin_yg_list[k]])

    j = 0
    for i in source_data[source_data['tagv_desc'] == '日照站出站流量检测'].index[1:-1]:
        filter_data.loc[j, '日照站出站流量检测'] = source_data.loc[i, 'tagv_value']
        for u in range(-4, 5):
            if (source_data.loc[i + u, 'tagv_desc'] in columns and abs(
                    source_data.loc[i, 'time_stamp'] - source_data.loc[i + u, 'time_stamp']) < time_delta):
                filter_data.loc[j, source_data.loc[i + u, 'tagv_desc']] = source_data.loc[i + u, 'tagv_value']
        j += 1
    filter_data['pressure_diff'] = filter_data['%s' % gongpin_yl_list[k * 2 + 1]] - filter_data[
        '%s' % gongpin_yl_list[k * 2]]

    filter_data2 = filter_data[(filter_data['日照站出站流量检测'] > 4000) & (filter_data['pressure_diff'] > 0.3)]
    del filter_data2['%s' % gongpin_yl_list[k * 2]]
    del filter_data2['%s' % gongpin_yl_list[k * 2 + 1]]

    filter_data2['%s' % gongpin_yg_list[k]] = filter_data2['%s' % gongpin_yg_list[k]].apply(
        lambda x: x * 1000 if x < 4 else x)
    filter_data2.dropna(inplace=True)
    display(filter_data2)

    if len(filter_data2) > 0:
        df = filter_data2.copy()
        df.columns = ['flow', 'power', 'pressure']

        # 假设你有一个名为df的DataFrame，包含两列数据：'flow' 和 'power'
        # 从DataFrame中提取数据
        flow_data = df['flow'].values
        power_data = df['power'].values

        a = do_predict()[k][0]
        b = do_predict()[k][1]
        c = do_predict()[k][2]


        # 定义 power 的计算公式
        def calculate_power(flow, w):
            pressure = a * pow(flow, 2) + b * flow + c
            return flow * pressure * w


        # 定义你要拟合的非线性函数
        def nonlinear_func(flow, w):
            return calculate_power(flow, w)


        # 使用curve_fit进行拟合
        params, covariance = curve_fit(nonlinear_func, flow_data, power_data)

        # params 包含了拟合后得到的参数值 w
        w_fit = params[0]

        # 打印拟合的参数值
        print(f"w: {w_fit}")

        # 计算拟合后的预测值
        predicted_power = nonlinear_func(flow_data, w_fit)

        # 绘制原始数据和拟合曲线
        plt.figure(figsize=(7, 3))
        plt.scatter(flow_data, power_data, label='Data', color='blue')
        plt.plot(flow_data, predicted_power, label=f'Fitted Power (w={w_fit:.4f})', color='red')
        plt.xlabel('Flow')
        plt.ylabel('Power')
        plt.legend()
        plt.title('Fitted Power vs Actual Power')
        plt.grid()
        plt.show()

        # 打印拟合完成的函数
        print(f"拟合后的函数: power = flow * ({w_fit:.6f}) * (%.12f* flow^2 + %.10f * flow + %.10f)" % (a, b, c))
    else:
        print('此泵数据太少，无法拟合！')


# ！！！！！同上，数据拟合函数不稳定，需要人工观察图像观察公式是否能够使用！！！！！！！
# ————————————————————————————变频泵power拟合——————————————————————
def var_do_predict():
    """
    返回变频泵的压力
    """
    return {0: [-2652460.2327716, -208.6294665, 0.0187294, -0.0000005],  # 日照变1
            1: [39.2999942, 0.0020471, -0.0000003, 0.0400509],  # 日照变2
            2: [1.6359, 0.00010057, -0.000000020706, 1],  # 日照变3
            3: [-1295.5442436, 0.1048064, -0.0000056, -0.0017940],  # 日照变4

            4: [167.3397374, -0.0017207, -0.0000003, 0.0110540],  # 东海变1
            5: [523567.3784744, -209.1970785, 0.0214270, 0.0001318],  # 东海变2
            6: [-187483.5966987, -107.4527942, 0.0150083, -0.0000047],  # 东海变3
            7: [37890018.3645441, -1820.0716376, 0.0116878, 0.0000001],  # 东海变4

            8: [-185793.4742081, -3.3157951, 0.0009063, -0.0000089],  # 淮安变1
            9: [-122246.6054944, -44.4958415, 0.0047474, -0.0000075],  # 淮安变2
            10: [220829.7789553, 11.2578268, -0.0020346, 0.0000072],  # 淮安变3
            11: [1.6359, 0.00010057, -0.000000020706, 1],  # 淮安变4

            12: [-276271.7589354, -12.8700326, 0.0029378, -0.0000064],  # 观音变1
            13: [-483.3216402, 0.1366338, -0.0000132, -0.0127857],  # 观音变2
            14: [66.5218345, 0.0014976, -0.0000004, 0.0272683],  # 观音变3
            15: [-9181152.7520526, 267.0006291, 0.0100057, -0.0000002]  # 观音变4
            }


# 原始拟合函数，对比用的
def calculate_power(flow, freq):
    return (((freq + 0.6) ** 3) * 395.6312 + ((freq + 0.6) ** 2) * 0.7192 * flow + (freq + 0.6) * (
        -5.2654) * 1e-5 * flow ** 2) * 1


# 画图生成数据
flow_range = np.linspace(3500, 4600, 100)  # flow取值范围
freq_range = np.linspace(0.1, 0.4, 100)  # freq取值范围
flow_mesh, freq_mesh = np.meshgrid(flow_range, freq_range)
power_mesh = calculate_power(flow_mesh, freq_mesh)
bianpin_yl_list = ['日仪增输日照站1P-7泵入口压力检测', '日仪增输日照站1P-7泵出口压力检测',
                   '日仪增输日照站1P-8泵入口压力检测', '日仪增输日照站1P-8泵出口压力检测'
    , '日仪增输日照站1P-9泵入口压力检测', '日仪增输日照站1P-9泵出口压力检测', '日仪增输日照站1P-10泵入口压力检测',
                   '日仪增输日照站1P-10泵出口压力检测'
    , '日仪增输东海站2P_3泵入口压力检测', '日仪增输东海站2P_3泵出口压力检测', '日仪增输东海站2P_4泵入口压力检测',
                   '日仪增输东海站2P_4泵出口压力检测'
    , '日仪增输东海站2P_5泵入口压力检测', '日仪增输东海站2P_5泵出口压力检测', '日仪增输东海站2P_6泵入口压力检测',
                   '日仪增输东海站2P_6泵出口压力检测'
    , '日仪增输淮安站3P_7泵入口压力检测', '日仪增输淮安站3P_7泵出口压力检测', '日仪增输淮安站3P_8泵入口压力检测',
                   '日仪增输淮安站3P_8泵出口压力检测'
    , '日仪增输淮安站3P_9泵入口压力检测', '日仪增输淮安站3P_9泵出口压力检测', '日仪增输淮安站3P_10泵入口压力检测',
                   '日仪增输淮安站3P_10泵出口压力检测'
    , '日仪增输观音站4P_3泵入口压力检测', '日仪增输观音站4P_3泵出口压力检测', '日仪增输观音站4P_4泵入口压力检测',
                   '日仪增输观音站4P_4泵出口压力检测'
    , '日仪增输观音站4P_5泵入口压力检测', '日仪增输观音站4P_5泵出口压力检测', '日仪增输观音站4P_6泵入口压力检测',
                   '日仪增输观音站4P_6泵出口压力检测']

bianpinqi_pinlv_list = ['日仪增输日照站1#变频器频率输入', '日仪增输日照站2#变频器频率输入'
    , '日仪增输东海站1#变频器输出频率检测', '日仪增输东海站2#变频器输出频率检测'
    , '日仪增输淮安站1#变频器输出频率检测', '日仪增输淮安站2#变频器输出频率检测'
    , '日仪增输观音站1#变频器输出频率检测', '日仪增输观音站2#变频器输出频率检测']

bianpin_yg_list = ['日仪增输日照站1P-7泵有功功率', '日仪增输日照站1P-8泵有功功率', '日仪增输日照站1P-9泵有功功率',
                   '日仪增输日照站1P-10泵有功功率'
    , '日仪增输东海站2P-3电机功率', '日仪增输东海站2P-4电机功率', '日仪增输东海站2P-5电机功率', '日仪增输东海站2P-6电机功率'
    , '日仪增输淮安站3P-7电机功率', '日仪增输淮安站3P-8电机功率', '日仪增输淮安站3P-9电机功率', '日仪增输淮安站3P-10电机功率'
    , '日仪增输观音站4P-3电机功率', '日仪增输观音站4P-4电机功率', '日仪增输观音站4P-5电机功率',
                   '日仪增输观音站4P-6电机功率']
time_delta = 60 * 8  # 时间差设置

m = 0
for k in range(8):
    if k % 2 == 0:
        m = k * 2
        bp1_in = bianpin_yl_list[k * 4]
        bp1_out = bianpin_yl_list[k * 4 + 1]
        bp2_in = bianpin_yl_list[k * 4 + 4]
        bp2_out = bianpin_yl_list[k * 4 + 5]
        bp_pl = bianpinqi_pinlv_list[k]
        bp1_yg = bianpin_yg_list[k * 2]
        bp2_yg = bianpin_yg_list[k * 2 + 2]

    else:
        m = k * 2 - 1
        bp1_in = bianpin_yl_list[k * 4 - 2]
        bp1_out = bianpin_yl_list[k * 4 - 1]
        bp2_in = bianpin_yl_list[k * 4 + 2]
        bp2_out = bianpin_yl_list[k * 4 + 3]
        bp_pl = bianpinqi_pinlv_list[k]
        bp1_yg = bianpin_yg_list[k * 2 - 1]
        bp2_yg = bianpin_yg_list[k * 2 + 1]
    sql = """
        select tagv_desc,tagv_value,tagv_fresh_time from  fz_tag_view
    where tagv_desc in ('{}','{}'
                       ,'{}','{}'
                       ,'日照站出站流量检测','{}','{}','{}') AND ( tagv_fresh_time LIKE '2023%' or tagv_fresh_time LIKE '2022%')
    ORDER BY tagv_fresh_time
        """.format(bp1_in, bp1_out, bp2_in, bp2_out, bp_pl, bp1_yg, bp2_yg)
    print(sql)
    conn = get_conn()
    result = {}
    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

    source_data = pd.DataFrame(data)
    source_data['time_stamp'] = source_data['tagv_fresh_time'].astype('str').apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())

    columns = ['%s' % bp1_in, '%s' % bp1_out
        , '%s' % bp2_in, '%s' % bp2_out
        , '日照站出站流量检测', '%s' % bp1_yg, '%s' % bp2_yg]
    filter_data = pd.DataFrame(columns=['%s' % bp1_in, '%s' % bp1_out
        , '%s' % bp2_in, '%s' % bp2_out
        , '日照站出站流量检测', '%s' % bp_pl, '%s' % bp1_yg, '%s' % bp2_yg])

    j = 0
    for i in source_data[source_data['tagv_desc'] == bp_pl].index[1:-1]:
        filter_data.loc[j, bp_pl] = source_data.loc[i, 'tagv_value']

        for u in range(-7, 7):
            if (source_data.loc[i + u, 'tagv_desc'] in columns and abs(
                    source_data.loc[i, 'time_stamp'] - source_data.loc[i + u, 'time_stamp']) < time_delta):
                filter_data.loc[j, source_data.loc[i + u, 'tagv_desc']] = source_data.loc[i + u, 'tagv_value']
        j += 1

    filter_data = filter_data[filter_data[bp_pl] > 20]
    filter_data = filter_data.dropna()

    filter_data['7泵压差'] = filter_data[bp1_out] - filter_data['%s' % bp1_in]
    filter_data['9泵压差'] = filter_data[bp2_out] - filter_data[bp2_in]
    filter_data = filter_data.loc[:, ['7泵压差', '9泵压差', '日照站出站流量检测', bp_pl, bp1_yg, bp2_yg]]

    columns = ['7泵压差', '日照站出站流量检测', bp_pl, bp1_yg]
    pump7 = pd.DataFrame(columns=columns)
    columns = ['9泵压差', '日照站出站流量检测', bp_pl, bp2_yg]
    pump9 = pd.DataFrame(columns=columns)

    for i in range(len(filter_data)):
        if filter_data.iloc[i, 0] > 0.3 and filter_data.iloc[i, 1] < 0.3:  # 开1不开2
            pump7 = pump7.append({'7泵压差': filter_data.iloc[i, 0], '日照站出站流量检测': filter_data.iloc[i, 2],
                                  bp_pl: filter_data.iloc[i, 3], bp1_yg: filter_data.iloc[i, 4]}, ignore_index=True)

        elif filter_data.iloc[i, 0] < 0.3 and filter_data.iloc[i, 1] > 0.3:  # 开2不开1
            pump9 = pump9.append({'9泵压差': filter_data.iloc[i, 1], '日照站出站流量检测': filter_data.iloc[i, 2],
                                  bp_pl: filter_data.iloc[i, 3], bp2_yg: filter_data.iloc[i, 5]}, ignore_index=True)

        elif filter_data.iloc[i, 0] > 0.3 and filter_data.iloc[i, 1] > 0.3:  # 都开
            if filter_data.iloc[i, 0] > filter_data.iloc[i, 1]:  # 大的是工频，小的是变频
                pump9 = pump9.append({'9泵压差': filter_data.iloc[i, 1], '日照站出站流量检测': filter_data.iloc[i, 2],
                                      bp_pl: filter_data.iloc[i, 3], bp2_yg: filter_data.iloc[i, 5]}, ignore_index=True)
            else:
                pump7 = pump7.append({'7泵压差': filter_data.iloc[i, 0], '日照站出站流量检测': filter_data.iloc[i, 2],
                                      bp_pl: filter_data.iloc[i, 3], bp1_yg: filter_data.iloc[i, 4]}, ignore_index=True)

    pump7 = pump7[pump7[bp1_yg] > 0]
    pump9 = pump9[pump9[bp2_yg] > 0]

    pump7[bp1_yg] = pump7[bp1_yg].apply(lambda x: x * 1000 if x < 4 else x)
    pump9[bp2_yg] = pump9[bp2_yg].apply(lambda x: x * 1000 if x < 4 else x)

    xunhuan_list = []
    # 出现次数最多的值貌似是填充值，直接删了.删错了也没事，整行删了少几个数而已
    if len(pump7) > 0:
        # print(pump7)
        # print(pump7[bp1_yg].value_counts())
        pump7 = pump7[pump7[bp1_yg] != pd.DataFrame(pump7[bp1_yg].value_counts()).index[0]]
    if len(pump9) > 0:
        # print(pump9)
        # print(pump9[bp2_yg].value_counts())
        pump9 = pump9[pump9[bp2_yg] != pd.DataFrame(pump9[bp2_yg].value_counts()).index[0]]

    t = 0
    for ee in [pump7, pump9]:

        print('开始拟合%s' % ee.columns[3][4:12])
        if len(ee) < 10:
            print('该泵数据太少，无法拟合！')
            continue
        df = ee.copy()
        df.columns = ['pressure', 'flow', 'freq', 'power']

        # 假设你有一个名为df的DataFrame，包含三列数据：'flow'、'freq' 和 'power'
        # 从DataFrame中提取数据
        flow_data = df['flow'].values
        df['freq'] = (df['freq'] - 30) / 50
        freq_data = df['freq'].values
        power_data = df['power'].values

        a = var_do_predict()[m + t][0]
        b = var_do_predict()[m + t][1]
        c = var_do_predict()[m + t][2]
        d = var_do_predict()[m + t][3]
        t += 2


        # print('a=',a,'b=',b,'c=',c,'d=',d)

        # 定义 pressure 的计算公式
        def calculate_pressure(freq, flow):
            return (((freq + 0.6) ** 2) * a + (freq + 0.6) * b * flow + c * pow(flow, 2)) * d


        # 定义 power 的计算公式
        def calculate_power(flow, freq, w):
            pressure = calculate_pressure(freq, flow)
            return flow * pressure * w


        # 定义你要拟合的非线性函数
        def nonlinear_func(data, w):
            flow, freq = data
            return calculate_power(flow, freq, w)


        # 使用curve_fit进行拟合
        params, covariance = curve_fit(nonlinear_func, (flow_data, freq_data), power_data)

        # params 包含了拟合后得到的参数值 w
        w_fit = params[0]

        # 生成拟合曲面的网格点
        flow_range = np.linspace(min(flow_data), max(flow_data), 100)
        freq_range = np.linspace(min(freq_data), max(freq_data), 100)
        flow_range, freq_range = np.meshgrid(flow_range, freq_range)

        # 计算拟合曲面上的点的 power 值
        power_fit = nonlinear_func((flow_range, freq_range), w_fit)

        # 绘制三维曲面
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(flow_data, freq_data, power_data, label='Data', color='blue')
        ax.plot_surface(flow_range, freq_range, power_fit, cmap='viridis', alpha=0.7,
                        label=f'Fitted Surface (w={w_fit:.4f})')
        ax.plot_surface(flow_mesh, freq_mesh, power_mesh, cmap='viridis_r', alpha=0.3)
        ax.set_xlabel('Flow')
        ax.set_ylabel('Freq')
        ax.set_zlabel('Power')
        plt.show()
        print(
            f"拟合后的函数: power = flow * (((pow((freq + 0.6),2)) * %.8f + (freq + 0.6) * %.8f * flow + %.8f * pow(flow, 2)) * %.8f) * ({w_fit:.8f})" % (
                a, b, c, d))
