import time
from pyomo.environ import *
import sys
import os
from os.path import dirname
import  numpy as np
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
warnings.filterwarnings("ignore")   #忽略警告
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['font.sans-serif']=['Simhei'] #显示中文
plt.rcParams['axes.unicode_minus']=False   #显示负号
from IPython.display import display
from base.mysql_conn import get_conn
conn = get_conn()



# 选择2022,2023年数据(站点出站压力、站点进站压力、流量数据)进行站间压差公式拟合
# △press压降 = a*出站压力 + b*flow^1.75 + c*flow + d

# return 4个站间的压降计算公式 （日照 - 东海，东海 - 淮安，淮安 - 观音，观音 - 仪征）
# ！！！！！！！！由于拟合可能出现误差，需要人工通过图像甄别公式的正确性！！！！！！！！！！！
# 公式不直接写入到主函数，由人工选择性填入。

out_in_press_code_list=['B_RYX_RZZ_PT1111','B_RYX_DHZ_PT2101','B_RYX_DHZ_PT2111','B_RYX_HAZ_PT3101','B_RYX_HAZ_PT3111','B_RYX_GYZ_PT4101','B_RYX_GYZ_PT4111','B_RYX_YZZ_PT5101']
out_in_press_name_list=['日照出压','东海进压','东海出压','淮安进压','淮安出压','观音进压','观音出压','仪征进压',]
for k in range(4):
    k=k*2
    sql = """
        select tagv_name,tagv_value,tagv_fresh_time from  fz_tag_view
    where tagv_name in ('{}','{}','B_RYX_RZZ_FT1111') AND tagv_fresh_time LIKE '2023-0%' and tagv_status = 0 and tagv_value > 0
    ORDER BY tagv_fresh_time
        """.format(out_in_press_code_list[k],out_in_press_code_list[k+1])
    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()
    source_data = pd.DataFrame(data)
    source_data['time_stamp'] = source_data['tagv_fresh_time'].astype('str').apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())

    source_data.replace({'tagv_name': {'B_RYX_RZZ_FT1111': '日照站出站流量',
                                '%s'%out_in_press_code_list[k]: '%s'%out_in_press_name_list[k],
                                '%s'%out_in_press_code_list[k+1]: '%s'%out_in_press_name_list[k+1]}}, inplace=True)


    columns = ['%s'%out_in_press_name_list[k],'%s'%out_in_press_name_list[k+1]]
    filter_data = pd.DataFrame(columns=['%s'%out_in_press_name_list[k],'%s'%out_in_press_name_list[k+1],'日照站出站流量'])

    time_delta = 15    # 时间差 / 秒
    j=0
    for i in source_data[source_data['tagv_name']=='日照站出站流量'].index[1:-1]:
        filter_data.loc[j,'日照站出站流量'] = source_data.loc[i,'tagv_value']
        for u in range(-2,3):
            if (source_data.loc[i+u,'tagv_name'] in columns and abs(source_data.loc[i,'time_stamp']-source_data.loc[i+u,'time_stamp'])<time_delta):
                filter_data.loc[j,source_data.loc[i+u,'tagv_name']] =  source_data.loc[i+u,'tagv_value']
        j+=1


    filter_data['管段压差'] = filter_data['%s'%out_in_press_name_list[k]] - filter_data['%s'%out_in_press_name_list[k+1]]
    del filter_data['%s'%out_in_press_name_list[k+1]]


    filter_data=filter_data[(filter_data['日照站出站流量']>2000)& (filter_data['日照站出站流量']<6000)]
    filter_data=filter_data[filter_data['日照站出站流量']!=pd.DataFrame(filter_data['日照站出站流量'].value_counts()).index[0]]
    filter_data.dropna(inplace=True)

    if len(filter_data)>100:
        print('开始拟合%s-%s管段压差'%(out_in_press_name_list[k],out_in_press_name_list[k+1]))
        # 定义拟合函数
        def nonlinear_func(xy, a, b, c, d):
            x, y = xy
            return a * x + b * (y ** 1.75) + c * y + d

        # 提取数据并转换为浮点数
        x_data = filter_data['%s'%out_in_press_name_list[k]].astype(float).values
        y_data = filter_data['日照站出站流量'].astype(float).values
        z_data = filter_data['管段压差'].astype(float).values

        # 组合自变量数据
        xy_data = (x_data, y_data)

        # 进行非线性拟合
        params, covariance = curve_fit(nonlinear_func, xy_data, z_data)

        # 提取拟合后的系数
        a, b, c ,d= params

        # 打印拟合后的系数
        #print(f"a: {a}")
        #print(f"b: {b}")
        #print(f"c: {c}")
        #print(f"d: {d}")
        # 打印拟合的函数表达式
        fit_expression = f"z = {a} * x + {b} * (y^1.75) + {c} * y + {d}"
        print(f"拟合的函数表达式：{fit_expression}")

        # 生成平面点
        x_range = np.linspace(min(x_data), max(x_data), 100)
        y_range = np.linspace(min(y_data), max(y_data), 100)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        z_fit = nonlinear_func((x_grid, y_grid), a, b, c, d)

        def plot_3D1(elev=30,azim=30):
            ax = plt.subplot(projection="3d")
            ax.scatter(x_data, y_data, z_data, c='b', marker='o', label='实际数据')
            ax.plot_surface(x_grid, y_grid, z_fit, cmap='viridis', alpha=0.8, label='拟合平面')

            ax.view_init(elev=elev,azim=azim)

            ax.set_xlabel("出压")
            ax.set_ylabel("流量")
            ax.set_zlabel("压降")
            plt.show()
        from ipywidgets import interact,fixed
        interact(plot_3D1,elev=[0,10,20,30,60,90],azip=(-180,180))             #设置范围
        plt.show()
    else:
        print('数据量太少无法拟合')