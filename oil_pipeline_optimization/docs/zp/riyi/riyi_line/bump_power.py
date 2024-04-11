import numpy as np
import pandas as pd
import math
# from pyomo.environ import *
from base.mysql_conn import get_conn
# from riyi_line.info_to_json import get_real_bump_data,get_bump_state

# 泵实际功率
def real_bumpp_power():
    """
    查询来获取泵的实际功率数据
    :return:
    """
    def get_bump_state():
        """
        获取泵的状态
        :return:
        """
        conn = get_conn()

        sql = """
        SELECT
     id,
     p_fz01 b_name,
     p_flag state,
    IF
     (
      ( SELECT TAGC_FZ08 FROM fz_tag_config WHERE tagc_fz03 = fb.id AND tagc_fz07 = 'BENG_2' ) - 
      ( SELECT TAGC_FZ08 FROM fz_tag_config WHERE tagc_fz03 = fb.id AND tagc_fz07 = 'BENG_1' ) > 0.3,
      1,0 
     ) is_open,
     fk_station as station,
     fb.p_linetype 
    FROM
     fz_beng fb 
    WHERE
     p_type = 1
     AND p_bengtype = 'B_TYPE_0' 
    ORDER BY
     p_fz02
        """
        bump_state = {}

        with conn.cursor() as cursor:
            cursor.execute(sql)
            data = cursor.fetchall()

            for elem in data:
                bump_state[elem['id']] = int(elem['is_open'])

        conn.close()

        return bump_state

    def get_real_bump_data():
        """
        获取泵的实际功率
        :return:
        """
        conn = get_conn()

        sql = """SELECT id,p_fz01 name,p_flag state,IFNULL((select SUM(TAGC_FZ08) 
        from fz_tag_config
        where tagc_fz03 = fz_beng.id and tagc_fz07='BENG_5'),0) power,
        fk_station station 
        FROM fz_beng where p_type = 1 and p_bengtype ='B_TYPE_0' 
        order by station,id"""

        power = []
        state = []
        id = []
        name = []
        station = []

        with conn.cursor() as cursor:
            cursor.execute(sql)
            data = cursor.fetchall()

            for elem in data:
                power.append(elem['power'])
                state.append(elem['state'])
                id.append(elem['id'])
                name.append(elem['name'])
                station.append(elem['station'])

        bump_state = get_bump_state()
        df = pd.DataFrame.from_dict({'power': power,
                                     'state': state,
                                     'id': id,
                                     'name': name,
                                     'station': station})

        df['state'] = df['id'].apply(lambda x: bump_state[x])
        df['power'] = df['state'] * df['power']

        def adjust_power(row):
            state = row['state']
            power = row['power']
            if (state == 1 and power <= 0):
                row['power'] = 2400.0

            return row

        df = df.apply(lambda row: adjust_power(row), axis=1)

        conn.close()
        return df
    real_power = list(get_real_bump_data().iloc[:, 0])
    real_power = pd.DataFrame(np.array(real_power).reshape(4, 6))
    real_power.iloc[3, :] = real_power.iloc[3, :].apply(lambda x: x * 1000 if 0 < x < 1000 else x)

    return real_power


#工频泵功率
# def predict_fixed_frequency_power(flow):
#     return 395.6312 + \
#            0.7192 * flow + \
#            -5.2654 * math.pow(10,-5) * math.pow(flow, 2)

# def pump_gp_power(flow):
#     """
#     8个工频泵有功计算公式。编号从0——7，每两个为一个站点的两个工频泵序号，以此类推。
#     1、power = flow * pressure  * w 拟合公式.
#     2、其中pressure也是拟合的公式。pressure = f(flow) = a*flow^2 + b*flow + c. 有pressure的拟合函数
#     return 带入flow计算的7个工频泵功率值
#     """
#     return {0: flow * (0.33725175635761406) * (
#                 -0.000000094675 * pow(flow, 2) + 0.0007256603 * flow + 0.3259694955),
#             1: flow * (0.33145380509258104) * (
#                         -0.000000005331 * pow(flow, 2) + -0.0000271323 * flow + 1.9025624965),
#             2: flow * (0.36600262052566543) * (-2.0706e-8 * pow(flow, 2) + 1.0057e-4 * flow + 1.6359),
#             3: flow * (0.3500262052566543) * (-2.0706e-8 * pow(flow, 2) + 1.0057e-4 * flow + 1.6359),
#             4: flow * (0.33112134816507) * (
#                         -0.000000044731 * pow(flow, 2) + 0.0002942832 * flow + 1.2588246840),
#             5: flow * (0.3426978802487336) * (
#                         -0.000000020706 * pow(flow, 2) + 0.0001005700 * flow + 1.6359),
#             6: flow * (0.33249882021238303) * (
#                         -0.000000041413 * pow(flow, 2) + 0.0002705021 * flow + 1.2816978385),
#             7: flow * (0.3388542396970702) * (
#                         -0.000000056491 * pow(flow, 2) + 0.0004074756 * flow + 0.9866817504)
#             }


#变频泵功率，常数项在工频泵的功率基础上和频率的立方成正比
# def predict_variable_frequency_power(flow_capacity,freq):
#
#     return (((freq+0.6)**3)*395.6312 + \
#            ((freq+0.6)**2)*0.7192*flow_capacity + \
#            (freq+0.6)*(-5.2654)*math.pow(10,-5)*math.pow(flow_capacity,2))*1.05

# def predict_variable_frequency_power(flow_capacity,freq):
#
#     return (((freq+0.6)**3)*395.6312 + \
#            ((freq+0.6)**2)*0.7192*flow_capacity + \
#            (freq+0.6)*(-5.2654)*math.pow(10,-5)*math.pow(flow_capacity,2))*1

# def predict_variable_frequency_power(flow_capacity,freq):
#
#     return -3981.58590194 * (freq + 0.6)**2 + 2.88652297 * (freq + 0.6) * flow_capacity + -0.00033455 * flow_capacity**2

# def pump_bp_power(flow,freq):
#     """
#     16个变频泵有功计算公式。每4个为一个站点的变频泵序号，以此类推。
#     1、power = flow * pressure  * w 拟合公式
#     2、pressure = a*(freq+0.6)^2 + b*(freq+0.6)*flow + c*flow^2。有pressure的拟合函数
#     """
#     return {0: flow * (((pow((freq + 0.6), 2)) * -2652460.23277160 + (
#                 freq + 0.6) * -208.62946650 * flow + 0.01872940 * pow(flow, 2)) * -0.00000050) * (
#                     0.35880283),
#             1: flow * (((pow((freq + 0.6), 2)) * 39.29999420 + (freq + 0.6) * 0.00204710 * flow + -0.00000030 * pow(
#                 flow, 2)) * 0.04005090) * (0.33032906),
#             2: flow * (((pow((freq + 0.6), 2)) * 1.63590000 + (freq + 0.6) * 0.00010057 * flow + -0.00000002 * pow(
#                 flow, 2)) * 1.00000000) * (0.36091030),
#             3: flow * (((pow((freq + 0.6), 2)) * -1295.54424360 + (freq + 0.6) * 0.10480640 * flow + -0.00000560 * pow(
#                 flow, 2)) * -0.00179400) * (0.36032962),
#
#             4: flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,
#                                                                                                                 -8) * pow(
#                 flow, 2)) * (0.34635286608847915) * 1.05,
#             5: flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,
#                                                                                                                 -8) * pow(
#                 flow, 2)) * (0.34635286608847915) * 1.05,
#             6: flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,
#                                                                                                                 -8) * pow(
#                 flow, 2)) * (0.34635286608847915) * 1.05,
#             7: flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,
#                                                                                                                 -8) * pow(
#                 flow, 2)) * (0.34635286608847915) * 1.05 ,
#
#             8: flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,
#                                                                                                                 -8) * pow(
#                 flow, 2)) * (0.34635286608847915) * 1.05 ,
#             9: flow * (((pow((freq + 0.6), 2)) * -122246.60549440 + (
#                         freq + 0.6) * -44.49584150 * flow + 0.00474740 * pow(flow, 2)) * -0.00000750) * (
#                     0.35017641),
#             10: flow * (((pow((freq + 0.6), 2)) * 220829.77895530 + (
#                         freq + 0.6) * 11.25782680 * flow + -0.00203460 * pow(flow, 2)) * 0.00000720) * (
#                      0.34447708),
#             11: flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,
#                                                                                                                  -8) * pow(
#                 flow, 2)) * (0.34635286608847915) * 1.05,
#
#             12: flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,
#                                                                                                                  -8) * pow(
#                 flow, 2)) * (0.34635286608847915) * 1.05,
#             13: flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,
#                                                                                                                  -8) * pow(
#                 flow, 2)) * (0.34635286608847915) * 1.05 ,
#             14: flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,
#                                                                                                                  -8) * pow(
#                 flow, 2)) * (0.34635286608847915) * 1.05 ,
#             15: flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,
#                                                                                                                  -8) * pow(
#                 flow, 2)) * (0.34635286608847915) * 1.05
#             }



def test(freq):
    print(freq,predict_variable_frequency_power(4134.272843589743,(freq-30.0)/50.0))


