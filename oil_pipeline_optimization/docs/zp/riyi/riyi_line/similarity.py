import numpy as np
import pandas as pd


from base.config import riyi_minutes, flow_minutes, pipe_diff_pressure_minutes
from base.mysql_conn import get_conn
from riyi_line.bump_choose import bump_choose
from riyi_line.bump_pressure import predict_fixed_frequency_bump_pressure_v1, regularize, \
    predict_var_frequency_bump_pressure_v1
from riyi_line.info_to_json import get_info_json
from riyi_line.input_to_json import get_input_json
from riyi_line.output_to_json import get_json_from_records
from riyi_line.station_settings import get_diff_pressure_between_stations, get_elec_price, get_flow_in, \
    predict_diff_pressure_between_stations
from utils.input import write_optimize_model_result_to_mysql, get_single_tag_value, get_last_time
history_find_model_result = [0]
history_find_model_elec_price = [0]


def history_find_model_re_glo():
    global history_find_model_result
    return history_find_model_result


def history_find_model_el_glo():
    global history_find_model_elec_price
    return history_find_model_elec_price


def get_history_flow(tag_name):
    last_time = get_last_time(tag_name)
    minute = 432000     # 300天时间
    conn = get_conn()

    sql = """
    select tagv_fresh_time, tagv_value from fz_tag_view t
where t.tagv_name = '{}'
and t.tagv_status = 0
and t.tagv_value > 0
and t.tagv_fresh_time > ('{}' - interval {} minute) 
    """.format(tag_name, last_time, minute)
    times = []
    values = []

    with conn.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()

        for elem in result:
            # t = elem['tagv_fresh_time'].strftime('%Y-%m-%d %H')  # 做天层面上的近似
            t = elem['tagv_fresh_time'].strftime('%Y-%m-%d')  # 做天层面上的近似
            times.append(str(t))
            v = float(elem['tagv_value'])  # 做100位上的近似
            v = v // 100 * 100
            values.append(v)

    conn.close()

    return times, values


# 获取历史时刻的站间压差
def get_history_di_pressure(similarity_time):
    # 连接数据库
    conn = get_conn()

    # 从数据库读取所有站点数据
    def get_data():
        sql = """
            SELECT a.tagv_name, a.tagv_value, a.tagv_fresh_time FROM work.fz_tag_view a
            WHERE a.tagv_status = '0' 
            AND a.tagv_value >= 0 
            AND a.tagv_name in ('B_RYX_RZZ_FT1111','B_RYX_RZZ_PT1111',
            'B_RYX_DHZ_PT2101','B_RYX_DHZ_PT2111',
            'B_RYX_HAZ_PT3101','B_RYX_HAZ_PT3111',
            'B_RYX_GYZ_PT4101',
            'B_RYX_GYZ_PT4111','B_RYX_YZZ_PT5101')
			AND a.tagv_fresh_time LIKE '{}%'
            """.format(similarity_time)  # 取站点列表中所有符合相似时间的对应点位数据
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
        t = elem['tagv_fresh_time'].strftime('%Y-%m-%d %H')
        # t = elem['tagv_fresh_time'].strftime('%Y-%m-%d')
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
    # rz_dh_time = drop_flow_outlier(rz_dh_time)
    # rz_dh_time.to_csv("rz_dh_time.csv")
    # print_json(rz_dh_time)
    rz_dh_time = np.array(rz_dh_time)
    rz_dh_time = rz_dh_time.tolist()
    for i in range(len(rz_dh_time)):  # 计算压差
        rz_dh_time[i].append(rz_dh_time[i][2] - rz_dh_time[i][3])
    # print(rz_dh_time.shape)

    dh_ha_time = dh_op_data.merge(ha_ip_data, on='time', how='inner')  # 东海-淮安
    dh_ha_time = rz_of_data.merge(dh_ha_time, on='time', how='inner')
    # dh_ha_time = drop_flow_outlier(dh_ha_time)
    dh_ha_time = np.array(dh_ha_time)
    dh_ha_time = dh_ha_time.tolist()
    for i in range(len(dh_ha_time)):  # 计算压差
        dh_ha_time[i].append(dh_ha_time[i][2] - dh_ha_time[i][3])
    # print(dh_ha_time.shape)

    ha_gy_time = ha_op_data.merge(gy_ip_data, on='time', how='inner')  # 淮安-观音
    ha_gy_time = rz_of_data.merge(ha_gy_time, on='time', how='inner')
    # ha_gy_time = drop_flow_outlier(ha_gy_time)
    ha_gy_time = np.array(ha_gy_time)
    ha_gy_time = ha_gy_time.tolist()
    for i in range(len(ha_gy_time)):  # 计算压差
        ha_gy_time[i].append(ha_gy_time[i][2] - ha_gy_time[i][3])
    # print(ha_gy_time.shape)

    gy_yz_time = gy_op_data.merge(yz_ip_data, on='time', how='inner')  # 观音-仪征
    gy_yz_time = rz_of_data.merge(gy_yz_time, on='time', how='inner')
    # gy_yz_time = drop_flow_outlier(gy_yz_time)
    gy_yz_time = np.array(gy_yz_time)
    gy_yz_time = gy_yz_time.tolist()
    for i in range(len(gy_yz_time)):  # 计算压差
        gy_yz_time[i].append(gy_yz_time[i][2] - gy_yz_time[i][3])
    # print(gy_yz_time.shape)
    conn.close()  # 关闭数据库连接
    return rz_dh_time, dh_ha_time, ha_gy_time, gy_yz_time


# 获取指定点位、指定时间的历史泵压力，用于计算开启个数
def get_history_open_bump_by_pressure(tag_name, time):
    conn = get_conn()

    sql = """
    select b.tagv_name, IFNULL(b.tagv_value,0) as tagValue 
    from  fz_tag_view b 
	WHERE b.tagv_name in ({})
	AND b.tagv_fresh_time LIKE '{}%'
	order by b.tagv_name, b.tagv_fresh_time
    """.format(tag_name, time)
    print('get_history_open_bump_by_pressure',sql)
    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        df = pd.DataFrame(list(data)).T
        if df.empty:  # 出现数据缺失
            return -1
        bump_data = list(df.loc['tagValue'])  # 取得的数据转为列表
        bump_data_name = list(df.loc['tagv_name'])

        data = []
        data_name = []
        for i in range(len(bump_data)):
            if bump_data_name[i] not in data_name:
                data_name.append(bump_data_name[i])
                data.append(bump_data[i])
        if len(data) < 12:  # 出现数据缺失
            return -1

        # 工频泵数据
        num_bumps = 6
        started_bumps = 0

        # fixed_freq_bump_data = bump_data.iloc[:,start_pos:start_pos + num_bumps * 2]
        for j in range(0, 12, 2):
            delta = data[j + 1] - data[j]
            if delta > 0.3:
                started_bumps += 1

    conn.close()
    return started_bumps


# 获取历史变频频率
def get_history_bump_freq(tag_name, time):
    conn = get_conn()

    sql = """
    select b.tagv_name, IFNULL(b.tagv_value,0) as tagValue 
    from  fz_tag_view b 
	WHERE b.tagv_name in ({})
	AND b.tagv_fresh_time LIKE '{}%'
	order by b.tagv_name, b.tagv_fresh_time
    """.format(tag_name, time)
    print('get_history_bump_freq',sql)
    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        df = pd.DataFrame(list(data)).T
        if df.empty:  # 出现数据缺失
            return 0

        bump_data = list(df.loc['tagValue'])  # 取得的数据转为列表
        bump_data_name = list(df.loc['tagv_name'])

        data = []
        data_name = []
        for i in range(len(bump_data)):
            if bump_data_name[i] not in data_name:
                data_name.append(bump_data_name[i])
                data.append(bump_data[i])
        # if len(data) < 2:  # 出现数据缺失
        #     return -1

        # 工频泵数据
        bump_freq = 0

        # fixed_freq_bump_data = bump_data.iloc[:,start_pos:start_pos + num_bumps * 2]
        for j in range(len(data)):
            if data[j] > 20:
                bump_freq = data[j]

    conn.close()
    # 返回如下格式{0: 5, 1: 5, 2: 4, 3: 5}
    return bump_freq


# 欧式距离公式计算
def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.power((a - b), 2)))


# 计算时间段内的欧式距离最小值
def calculate_euclidean(now, history):
    min_distance = 1000
    min_distance_time = '0'
    for i in range(len(history)):
        distance = euclidean_distance(now, history[i][4])
        if distance < min_distance:
            min_distance_time = history[i][0]
            min_distance = distance
    return min_distance, min_distance_time


# 获取对应时间的历史泵压差和频率，准备输入泵选择模型
def find_bump_choose(data):
    rz_time = data[1]
    dh_time = data[3]
    ha_time = data[5]
    gy_time = data[7]
    rz_bump_choose = get_history_open_bump_by_pressure("'ryx_RZZS_PT1905A','ryx_RZZS_PT1905B',"
                                                       "'ryx_RZZS_PT1906A','ryx_RZZS_PT1906B',"
                                                       "'ryx_RZZS_PT1907A','ryx_RZZS_PT1907B',"
                                                       "'ryx_RZZS_PT1908A','ryx_RZZS_PT1908B',"
                                                       "'ryx_RZZS_PT1909A','ryx_RZZS_PT1909B',"
                                                       "'ryx_RZZS_PT1910A','ryx_RZZS_PT1910B'", rz_time)

    dh_bump_choose = get_history_open_bump_by_pressure("'ryx_DHZS_PT2901A','ryx_DHZS_PT2901B',"
                                                       "'ryx_DHZS_PT2902A','ryx_DHZS_PT2902B',"
                                                       "'ryx_DHZS_PT2903A','ryx_DHZS_PT2903B',"
                                                       "'ryx_DHZS_PT2904A','ryx_DHZS_PT2904B',"
                                                       "'ryx_DHZS_PT2905A','ryx_DHZS_PT2905B',"
                                                       "'ryx_DHZS_PT2906A','ryx_DHZS_PT2906B'", dh_time)

    ha_bump_choose = get_history_open_bump_by_pressure("'ryx_HAZS_PT3905A','ryx_HAZS_PT3905B',"
                                                       "'ryx_HAZS_PT3906A','ryx_HAZS_PT3906B',"
                                                       "'ryx_HAZS_PT3907A','ryx_HAZS_PT3907B',"
                                                       "'ryx_HAZS_PT3908A','ryx_HAZS_PT3908B',"
                                                       "'ryx_HAZS_PT3909A','ryx_HAZS_PT3909B',"
                                                       "'ryx_HAZS_PT3910A','ryx_HAZS_PT3910B'", ha_time)

    gy_bump_choose = get_history_open_bump_by_pressure("'ryx_GYZS_PT4901A','ryx_GYZS_PT4901B',"
                                                       "'ryx_GYZS_PT4902A','ryx_GYZS_PT4902B',"
                                                       "'ryx_GYZS_PT4903A','ryx_GYZS_PT4903B',"
                                                       "'ryx_GYZS_PT4904A','ryx_GYZS_PT4904B',"
                                                       "'ryx_GYZS_PT4905A','ryx_GYZS_PT4905B',"
                                                       "'ryx_GYZS_PT4906A','ryx_GYZS_PT4906B'", gy_time)

    rz_freq = get_history_bump_freq("'ryx_RZZS_AI15001A','ryx_RZZS_AI15002A'", rz_time)
    dh_freq = get_history_bump_freq("'ryx_DHZS_AI25001A','ryx_DHZS_AI25002A'", dh_time)
    ha_freq = get_history_bump_freq("'ryx_HAZS_AI35001A','ryx_HAZS_AI35002A'", ha_time)
    gy_freq = get_history_bump_freq("'ryx_GYZS_AI45001A','ryx_GYZS_AI45002A'", gy_time)
    if (rz_freq == -1) | (dh_freq == -1) | (ha_freq == -1) | (gy_freq == -1) | (gy_bump_choose == -1) | (
            ha_bump_choose == -1) | (dh_bump_choose == -1) | (rz_bump_choose == -1):
        print("bump choose dont have enough time")
        print("rz_freq:", rz_freq)
        print("dh_freq:", dh_freq)
        print("ha_freq:", ha_freq)
        print("gy_freq:", gy_freq)
        print("gy_bump_choose:", gy_bump_choose)
        print("ha_bump_choose:", ha_bump_choose)
        print("dh_bump_choose:", dh_bump_choose)
        print("rz_bump_choose:", rz_bump_choose)
        return -1
    # xres = [[2, 41], [3, 0], [1, 41], [0, 32.9855]]
    if rz_freq != 0:
        rz_bump_choose -= 1
    if dh_freq != 0:
        dh_bump_choose -= 1
    if ha_freq != 0:
        ha_bump_choose -= 1
    if gy_freq != 0:
        gy_bump_choose -= 1
    # xres = [[2, 41], [3, 0], [1, 41], [0, 32.9855]]
    result = [[rz_bump_choose, rz_freq], [dh_bump_choose, dh_freq], [ha_bump_choose, ha_freq],
              [gy_bump_choose, gy_freq]]
    print(result)
    return result


# 计算水力坡降
def bump_choose_pojiang(flow, bump_choose_result, di_pressure_between_station):
    reg, initial_pressure = regularize(flow)  # 计算reg
    y = []
    bump_freq_after_calculate = []
    for i in range(len(bump_choose_result)):  # 计算是否开变频
        if bump_choose_result[i][1] != 0:
            y.append(1)
            if bump_choose_result[i][1] >= 30:
                temp = (bump_choose_result[i][1] - 30.0) / 50.0
                bump_freq_after_calculate.append(temp)
            else:
                temp = bump_choose_result[i][1] / 50
                bump_freq_after_calculate.append(temp)
        else:
            y.append(0)
            bump_freq_after_calculate.append(0)
    DiffPressureLasttime = get_diff_pressure_between_stations(minutes=pipe_diff_pressure_minutes)
    # 计算各站压降
    station0_outbound_pressure = predict_fixed_frequency_bump_pressure_v1(flow, reg[0], 0) * \
                                 bump_choose_result[0][0] + \
                                 predict_var_frequency_bump_pressure_v1(flow, bump_freq_after_calculate[0], reg[0], 0) * \
                                 y[0] + initial_pressure

    station1_inbound_pressure = station0_outbound_pressure - predict_diff_pressure_between_stations(flow, DiffPressureLasttime[0], 0)

    station1_outbound_pressure = station1_inbound_pressure + \
                                 predict_fixed_frequency_bump_pressure_v1(flow, reg[1], 1) * bump_choose_result[1][0] + \
                                 predict_var_frequency_bump_pressure_v1(flow, bump_freq_after_calculate[1], reg[1], 1) * \
                                 y[
                                     1]

    station2_inbound_pressure = station1_outbound_pressure - predict_diff_pressure_between_stations(flow, DiffPressureLasttime[1], 1)

    station2_outbound_pressure = station2_inbound_pressure + \
                                 predict_fixed_frequency_bump_pressure_v1(flow, reg[2], 2) * bump_choose_result[2][0] + \
                                 predict_var_frequency_bump_pressure_v1(flow, bump_freq_after_calculate[2], reg[2], 2) * \
                                 y[
                                     2]

    station3_inbound_pressure = station2_outbound_pressure - predict_diff_pressure_between_stations(flow, DiffPressureLasttime[2], 2)

    station3_outbound_pressure = station3_inbound_pressure + \
                                 predict_fixed_frequency_bump_pressure_v1(flow, reg[3], 3) * bump_choose_result[3][0] + \
                                 predict_var_frequency_bump_pressure_v1(flow, bump_freq_after_calculate[3], reg[3], 3) * \
                                 y[
                                     3]

    station4_inbound_pressure = station3_outbound_pressure - predict_diff_pressure_between_stations(flow, DiffPressureLasttime[3], 3)

    stations_pressure = [station0_outbound_pressure,
                         station1_inbound_pressure,
                         station1_outbound_pressure, station2_inbound_pressure,
                         station2_outbound_pressure, station3_inbound_pressure,
                         station3_outbound_pressure, station4_inbound_pressure]
    choice = bump_choose(bump_choose_result, flow, stations_pressure)
    # print(choice)
    return choice


def similarity_main(flow):
    history_flow_time, history_flow = get_history_flow('B_RYX_RZZ_FT1111')  # 获取流量点位的历史数据和时间
    print("get history flow",history_flow)
    if len(history_flow_time) == 0:
        print("get history flow failure")
        return 0
    now_flow = flow // 100 * 100  # 对流量值做100左右的近似
    simility_time = []
    for i in range(len(history_flow_time)):
        if (now_flow == history_flow[i]) & (history_flow_time[i] not in simility_time):  # 选取100左右相同且没有记录过的时间
            simility_time.append(history_flow_time[i])
    simility_time_number = len(simility_time)
    print("get simility flow time")
    if simility_time_number == 0:
        print("similar model don‘t find similar time")
        return 0
    else:
        print("similar model get enough similar time")


    now_di_pressure = get_diff_pressure_between_stations(riyi_minutes)  # 当前时刻站间压差


    distance_result = []
    min_1 = 1000  # 最相似记录与坐标
    min_1_index = -1
    for i in range(simility_time_number):  # 对于每条相似时间
        rz_dh_time, dh_ha_time, ha_gy_time, gy_yz_time = get_history_di_pressure(simility_time[i])  # 获取时间段内历史压差

        per_distance_result = [-1, '0', -1, '0', -1, '0', -1, '0', 1]
        # 获取每个站点时间段内的欧式距离最小
        per_distance_result[0], per_distance_result[1] = calculate_euclidean(now_di_pressure[0], rz_dh_time)
        per_distance_result[2], per_distance_result[3] = calculate_euclidean(now_di_pressure[1], dh_ha_time)
        per_distance_result[4], per_distance_result[5] = calculate_euclidean(now_di_pressure[2], ha_gy_time)
        per_distance_result[6], per_distance_result[7] = calculate_euclidean(now_di_pressure[3], gy_yz_time)
        per_distance_result[8] = per_distance_result[0] + per_distance_result[2] + \
                                 per_distance_result[4] + per_distance_result[6]  # 计算所有欧式距离的和
        if min_1 > per_distance_result[8]:
            min_1 = per_distance_result[8]  # 找到最小欧式距离和及其坐标
            min_1_index = i
        distance_result.append(per_distance_result)  # 每行是一条记录，前8列是对应时间段内各站间压差的距离和时间，9列为欧式距离的和
    min_2 = 1000
    min_2_index = -1
    for i in range(simility_time_number):
        if (min_2 > distance_result[i][8]) & (i != min_1_index):  # 找到不等于最小值的第二最小值
            min_2 = distance_result[i][8]
            min_2_index = i
    print("min_1_index", min_1_index)
    print("min_2_index", min_2_index)
    flag1 = 1
    flag2 = 1
    if min_1_index == -1:  # 不存在最小点
        print("dont find min_1")
        return 0
    else:
        min_1_data = distance_result[min_1_index]
        bump_choose_min_1 = find_bump_choose(min_1_data)
        if bump_choose_min_1 != -1:  # 数据不缺失
            choice_1 = bump_choose_pojiang(flow, bump_choose_min_1, min_1_data)
        else:
            print("bump choose dont have enough data in min1")
            flag1 = 0
    if min_2_index != -1:  # 不缺
        min_2_data = distance_result[min_2_index]
        bump_choose_min_2 = find_bump_choose(min_2_data)
        if bump_choose_min_2 != -1:  # 数据不缺失
            choice_2 = bump_choose_pojiang(flow, bump_choose_min_2, min_2_data)
        else:
            print("bump choose dont have enough data in min2")
            flag2 = 0
    else:
        bump_choose_min_2 = -1
    # xres = [[2, 41], [3, 0], [1, 41], [0, 32.9855]]
    if (bump_choose_min_1 == -1) & (bump_choose_min_2 == -1):
        print("two history choice all dont have enough data")
        return 0
    if (flag1 != 0) & (flag2 != 0):  # 都不缺失
        if choice_1[0][9] <= choice_2[0][9]:
            choice_result = choice_1
            print("choose choice_1")
        else:
            choice_result = choice_2
            print("choose choice_2")
    elif flag1 != 0:  # min2缺失
        choice_result = choice_1
        print("choose choice_1")
    elif flag2 != 0:  # min1缺失
        choice_result = choice_2
        print("choose choice_2")
    else:       # 都缺
        print("dont choose any one")
        return 0
    if choice_result[0][9] > choice_result[1][9]:
        print("history power is bigger than now")
        return 0
    else:
        print("optimize success")
        print(choice_result)
        elec_price = get_elec_price()
        from riyi_line.station_settings import glo
        riyi_success_flag = glo()
        global history_find_model_result
        global history_find_model_elec_price
        # print("flag!!!", yichang_success_flag)
        if (riyi_success_flag != 1):
            print("optimize failed as data problem")
            return 0
        else:
            history_find_model_result = choice_result
            history_find_model_elec_price = elec_price
            # write_optimize_model_result_to_mysql(2, get_input_json(), get_info_json(),
            #                                      get_json_from_records(choice_result, elec_price))
            return 1


if __name__ == "__main__":
    flow = get_flow_in(minutes=flow_minutes)  # 取指定时长流量均值
    print("flow:", flow)
    similarity_main(flow)
