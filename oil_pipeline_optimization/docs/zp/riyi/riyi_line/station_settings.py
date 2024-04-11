import pandas as pd
import numpy as np

riyi_success_flag_station = 1
from base.config import *
from base.mysql_conn import get_conn
from entity.line import Line
from utils.input import get_single_tag_value, get_last_tag_value, get_single_tag_value_config

# from riyi_line.ry_forecast_di_pressure import RegressDiPressure

# diff_pressure_predictor = RegressDiPressure()
riyi = Line(line_id=1)


def glo():
    global riyi_success_flag_station
    return riyi_success_flag_station


# def get_flow_in_from_csv():
#     path = os.path.abspath(os.path.dirname(sys.argv[0]))
#     flow = pd.read_csv(os.path.join(path,'流量压力.csv'), header=0, encoding='utf-8')
#     return np.mean(flow['日照站出站流量'])

# def get_diff_pressure_between_stations_from_csv():
#     path = os.path.abspath(os.path.dirname(sys.argv[0]))
#     flow = pd.read_csv(os.path.join(path,'流量压力.csv'),header=0,encoding='utf-8')
#     result = {}
#     result[0] = np.mean(flow['日照站出站压力']) - np.mean(flow['东海站进站压力'])
#
#     result[1] = np.mean(flow['东海站出站压力']) - np.mean(flow['淮安站进站压力'])
#
#     result[2] = np.mean(flow['淮安站出站压力']) - np.mean(flow['观音站进站压力'])
#
#     result[3] = np.mean(flow['观音站出站压力']) - np.mean(flow['仪征站进站压力'])
#
#     return result

def get_diff_pressure_between_stations_config():  # ！！！得到不同站之间的压降
    global riyi_success_flag_station
    result = {}
    result[0] = np.mean(get_single_tag_value_config('B_RYX_RZZ_PT1111')['value']) - \
                np.mean(get_single_tag_value_config('B_RYX_DHZ_PT2101')['value'])

    result[1] = np.mean(get_single_tag_value_config('B_RYX_DHZ_PT2111')['value']) - \
                np.mean(get_single_tag_value_config('B_RYX_HAZ_PT3101')['value'])

    result[2] = np.mean(get_single_tag_value_config('B_RYX_HAZ_PT3111')['value']) - \
                np.mean(get_single_tag_value_config('B_RYX_GYZ_PT4101')['value'])

    result[3] = np.mean(get_single_tag_value_config('B_RYX_GYZ_PT4111')['value']) - \
                np.mean(get_single_tag_value_config('B_RYX_YZZ_PT5101')['value'])
    if np.isnan(result[0]):
        result[0] = np.mean(0)
        print("use 0 diff pressure riyi 0")
    if np.isnan(result[1]):
        result[1] = np.mean(0)
        print("use 0 diff pressure riyi 1")
    if np.isnan(result[2]):
        result[2] = np.mean(0)
        print("use 0 diff pressure riyi 2")
    if np.isnan(result[3]):
        result[3] = np.mean(0)
        print("use 0 diff pressure riyi 3")
    # print('get_diff_pressure_between_stations_result', result)
    return result


#
# def get_diff_pressure_between_stations(minutes=riyi_minutes):
#     global riyi_success_flag_station
#     #flow = pd.read_csv('流量压力.csv',header=0,encoding='GBK')
#     result = {}
#     result[0] = np.mean(get_single_tag_value('B_RYX_RZZ_PT1111',minutes)['value']) - \
#                 np.mean(get_single_tag_value('B_RYX_DHZ_PT2101',minutes)['value'])
#
#     result[1] = np.mean(get_single_tag_value('B_RYX_DHZ_PT2111',minutes)['value']) - \
#                 np.mean(get_single_tag_value('B_RYX_HAZ_PT3101',minutes)['value'])
#
#     result[2] = np.mean(get_single_tag_value('B_RYX_HAZ_PT3111',minutes)['value']) - \
#                 np.mean(get_single_tag_value('B_RYX_GYZ_PT4101',minutes)['value'])
#
#     result[3] = np.mean(get_single_tag_value('B_RYX_GYZ_PT4111',minutes)['value']) - \
#                 np.mean(get_single_tag_value('B_RYX_YZZ_PT5101',minutes)['value'])
#
#     # if np.isnan(result[0]):
#     #     result[0] = np.mean(get_single_tag_value('B_RYX_RZZ_PT1111', long_riyi_minutes)['value']) - \
#     #                 np.mean(get_single_tag_value('B_RYX_DHZ_PT2101', long_riyi_minutes)['value'])
#     #     print("use long diff pressure riyi 0")
#     #     riyi_success_flag_station = 0
#     # if np.isnan(result[1]):
#     #     result[1] = np.mean(get_single_tag_value('B_RYX_DHZ_PT2111', long_riyi_minutes)['value']) - \
#     #                 np.mean(get_single_tag_value('B_RYX_HAZ_PT3101', long_riyi_minutes)['value'])
#     #     print("use long diff pressure riyi 1")
#     #     riyi_success_flag_station = 0
#     # if np.isnan(result[2]):
#     #     result[2] = np.mean(get_single_tag_value('B_RYX_HAZ_PT3111', long_riyi_minutes)['value']) - \
#     #                 np.mean(get_single_tag_value('B_RYX_GYZ_PT4101', long_riyi_minutes)['value'])
#     #     print("use long diff pressure riyi 2")
#     #     riyi_success_flag_station = 0
#     # if np.isnan(result[3]):
#     #     result[3] = np.mean(get_single_tag_value('B_RYX_GYZ_PT4111', long_riyi_minutes)['value']) - \
#     #                 np.mean(get_single_tag_value('B_RYX_YZZ_PT5101', long_riyi_minutes)['value'])
#     #     print("use long diff pressure riyi 3")
#     #     riyi_success_flag_station = 0
#     if np.isnan(result[0]):
#         result[0] = np.mean(0)
#         print("use 0 diff pressure riyi 0")
#     if np.isnan(result[1]):
#         result[1] = np.mean(0)
#         print("use 0 diff pressure riyi 1")
#     if np.isnan(result[2]):
#         result[2] = np.mean(0)
#         print("use 0 diff pressure riyi 2")
#     if np.isnan(result[3]):
#         result[3] = np.mean(0)
#         print("use 0 diff pressure riyi 3")
#     print('get_diff_pressure_between_stations_result',result)
#     return result

def get_last_in_pressure(minutes=riyi_minutes):  # ！！！获得最新时间段内的站点压力值
    # flow = pd.read_csv('流量压力.csv',header=0,encoding='GBK')
    result = {}
    global riyi_success_flag_station
    result[0] = 0.63

    result[1] = np.mean(get_single_tag_value('B_RYX_DHZ_PT2101', minutes)['value'])  # 得到对应站点指定时间段最新流量，算平均值
    # print('111',get_single_tag_value('B_RYX_DHZ_PT2101',minutes)['value'])
    result[2] = np.mean(get_single_tag_value('B_RYX_HAZ_PT3101', minutes)['value'])

    result[3] = np.mean(get_single_tag_value('B_RYX_GYZ_PT4101', minutes)['value'])

    # if np.isnan(result[1]):
    #     result[1] = np.mean(get_single_tag_value('B_RYX_DHZ_PT2101',long_riyi_minutes)['value'])
    #     print("use long last in pressure riyi 1")
    #     riyi_success_flag_station = 0
    # if np.isnan(result[2]):
    #     result[2] = np.mean(get_single_tag_value('B_RYX_HAZ_PT3101',long_riyi_minutes)['value'])
    #     print("use long last in pressure riyi 2")
    #     riyi_success_flag_station = 0
    # if np.isnan(result[3]):
    #     result[3] = np.mean(get_single_tag_value('B_RYX_GYZ_PT4101',long_riyi_minutes)['value'])
    #     print("use long last in pressure riyi 3")
    #     riyi_success_flag_station = 0
    if np.isnan(result[1]):
        result[1] = np.mean(0)
        print("use 0 in pressure riyi 1")
    if np.isnan(result[2]):
        result[2] = np.mean(0)
        print("use 0 in pressure riyi 2")
    if np.isnan(result[3]):
        result[3] = np.mean(0)
        print("use 0 in pressure riyi 3")

    print('get_last_in_pressure_result', result)
    return result


# 获取流量从csv，获取压差从csv，
# 获取两站间压差，获取上一时间段压差，获取流量，预测两站间压差，获取电价，获取最大/小出/入站压力
def get_flow_in(minutes=riyi_minutes):
    # !!!这个函数的主要目的是获取特定标签在一段时间内的流量数据，并根据一系列条件对流量数据进行处理和分析，
    # !!!最终返回处理后的流量平均值。如果流量数据为空或变化过大，函数会采取一些特定的措施，如使用固定值或最后一条非零数据作为替代。
    # flow = pd.read_csv('流量压力.csv', header=0, encoding='GBK')
    global riyi_success_flag_station
    flow = get_single_tag_value('B_RYX_RZZ_FT1111', minutes)
    if flow.empty:
        # flow = get_last_tag_value('B_RYX_RZZ_FT1111')  # 如果出现空值异常，取0
        flow = 9.5
        print("use 9.5 in riyi flow")
        riyi_success_flag_station = 0
        return np.mean(flow)
    flow_array = np.array(flow)
    flow_list = flow_array.tolist()
    flow_list_big_than_10 = []
    MIN = 10000
    MAX = 0
    for i in flow_list:
        if i[0] > 10:
            flow_list_big_than_10.append(i[0])  # 留大于10的流量数据
            if i[0] < MIN:
                MIN = i[0]
            if i[0] > MAX:
                MAX = i[0]
    mean = float(np.mean(flow_list_big_than_10))
    mean_max = mean * 1.2
    mean_min = mean * 0.8
    if (MAX > mean_max) | (MIN < mean_min) | (len(flow_list_big_than_10) == 0):  # 流量变化过大或全部过小
        if len(flow_list_big_than_10) == 0:
            # flow = get_last_tag_value('B_RYX_RZZ_FT1111')  # 如果出现空值异常，取1
            flow = 9.5
            print("use 9.5 in riyi flow")
            riyi_success_flag_station = 0
            return np.mean(flow)
        flow = get_last_tag_value('B_RYX_RZZ_FT1111')  # 如果出现流量变化过大，说明正在启停输，取倒数第一条非零数据
        print("riyi flow is change to big in 2 hour or small than 10")
        # riyi_success_flag_station = 0
        return np.mean(flow['value'])
    else:
        return np.mean(flow_list_big_than_10)


def get_config_flow():
    """
    查询日照站出站流量检测
    :return:
    """
    conn = get_conn()
    sql = """
            SELECT tagc_fz08 FROM `fz_tag_config`
        where tagc_name ='B_RYX_RZZ_FT1111'
            """
    with conn.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()
    result = pd.DataFrame(result)

    return result


# def predict_diff_pressure_between_stations(flow_capacity,diff_pressure_last_time,index):
#     diff = diff_pressure_predictor.forecast_di_pressure(flow_capacity,diff_pressure_last_time,index)
#     return diff

# def fitting_pressure_drop(out_press,flow):
#     """
#     # 站间压降拟合公式
#     """
#     return {0:0.23279718602698526 * out_press + -1.202584643665587e-06 * pow(flow,1.75) + 0.0017658793380332714 * flow + -3.10853570222897,  # 日照出 - 东海进
#             1:0.34118400052032843 * out_press + 1.0220610620028407e-06 * pow(flow,1.75) + -0.00041556337484768775 * flow + 0.8934046804856154, # 东海出 - 淮安进
#             2:0.4258903967608222 * out_press + 7.513171533083859e-07 * pow(flow,1.75) + -0.00010291345426133052 * flow + -0.19361199027722945, # 淮安出 - 观音进
#             3:0.23279718602698526 * out_press + -1.202584643665587e-06 * pow(flow,1.75) + 0.0017658793380332714 * flow + -2.40853570222897  # 观音出 - 仪征进
#     }
# def pressure_drop(out_press,flow,index):
#     return fitting_pressure_drop(out_press,flow)[index]


def get_station_pipeline_mapping():
    return {1: [0], 2: [1], 3: [2], 4: [3]}


station_pipeline_mapping = get_station_pipeline_mapping()


def get_elec_price():
    result = {}
    origin = riyi.get_elec_price()
    for s_id, value in origin.items():
        for p in station_pipeline_mapping[s_id]:
            result[p] = value
    print('get_elec_price_result', result)
    return result


def get_max_outbound_pressure():
    result = {}
    origin = riyi.get_max_outbound_pressure()
    for s_id, value in origin.items():
        for p in station_pipeline_mapping[s_id]:
            result[p] = value
    print('get_max_outbound_pressure_result', result)
    return result


def get_max_inbound_pressure():
    result = {}
    origin = riyi.get_max_inbound_pressure()
    for s_id, value in origin.items():
        for p in station_pipeline_mapping[s_id]:
            result[p] = value
    print("get_max_inbound_pressure_result", result)
    return result


def get_min_inbound_pressure():
    result = {}
    origin = riyi.get_min_inbound_pressure()
    for s_id, value in origin.items():
        for p in station_pipeline_mapping[s_id]:
            result[p] = value
    print('get_min_inbound_pressure_result', result)
    return result


def get_next_min_inbound_pressure():
    """
    查询途径和终点所有站点的最小流入高度
    """
    conn = get_conn()

    sql = """SELECT v.station_low station_low
    FROM fz_station_line as  s  LEFT JOIN fz_station v on  s.station_id= v.station_id 
    where s.line_id = 1 and s.sl_type in ('oil','terminal')
    order by s.pipeline_id"""
    with conn.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()
    result = pd.DataFrame(result)
    result.iloc[:, 0] = result.iloc[:, 0].astype('float')
    result = result.iloc[:, 0].to_dict()
    print('get_next_min_inbound_pressure', result)
    return result


def get_next_max_inbound_pressure():
    """
    查询途径和终点所有站点的最大进站压力
    """
    conn = get_conn()

    sql = """SELECT v.station_hight station_high
    FROM fz_station_line as  s  LEFT JOIN fz_station v on  s.station_id= v.station_id 
    where s.line_id = 1 and s.sl_type in ('oil','terminal')
    order by s.pipeline_id"""
    with conn.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()
    result = pd.DataFrame(result)
    result.iloc[:, 0] = result.iloc[:, 0].astype('float')
    result = result.iloc[:, 0].to_dict()
    print('get_next_max_inbound_pressure', result)
    return result


'''
print(get_min_inbound_pressure())

print(get_elec_price())

print(get_max_outbound_pressure())

print(get_max_inbound_pressure())

print(get_last_in_pressure(minutes=60*24))
'''
