import numpy as np
import pandas as pd

from base.mysql_conn import get_conn
from riyi_line.riyi_bump_pressure_model import riyi_bump_pressure_model
from riyi_line.bump_choose import get_bumps_data_mysql
# pressure_model = riyi_bump_pressure_model()


def do_predict_1(flow):
    return 1.6359 + \
           1.0057 * pow(10,-4) * flow + \
           -2.0706 * pow(10,-8) * pow(flow, 2)
def do_predict(flow):
    return {0: -0.000000094675 * flow ** 2 + 0.0007256603 * flow + 0.3259694955,  # 日照1
            1: -0.000000005331 * flow ** 2 + -0.0000271323 * flow + 1.9025624965,  # 日照2
            2: -0.000000019866 * flow ** 2 + 0.0000678761 * flow + 1.7679022152,  # 东海1
            3: -0.000000108118 * flow ** 2 + 0.0008077043 * flow + 0.2038966221,  # 东海2
            4: -0.000000041413 * flow ** 2 + 0.0002705021 * flow + 1.2816978385,  # 淮安1
            5: -0.000000056491 * flow ** 2 + 0.0004074756 * flow + 0.9866817504,  # 淮安2
            6: -0.000000044731 * flow ** 2 + 0.0002942832 * flow + 1.2588246840,  # 观音1
            7: -0.000000020706 * flow ** 2 + 0.0001005700 * flow + 1.6359  # 观音2
            }

#工频泵，所有站的预测系数一样，不同点在于reg项
# def predict_fixed_frequency_bump_pressure_v1(flow, regularization, index):
#     pressure = do_predict(flow) * (1 + regularization)
#     return pressure
#工频泵flow--press * 调整系数
def predict_fixed_frequency_bump_pressure_v1(flow, regularization, index):
    """
    这个函数用于计算工频泵的预测压力。
    :param flow: 流量
    :param regularization: 正则化项
    :param index: 站点索引
    :return: 返回预测的压力
    """
    pressure = do_predict(flow)[index]
    return pressure

#变频泵，所有站的预测系数一样，不同点在于reg项
# def predict_var_frequency_bump_pressure_v1(flow, freq, regularization, index):
#     #started = 1 if (freq>math.exp(-10)) else 0
#     return (((freq+0.6)**2) * 1.6359 + \
#             (freq+0.6) * 1.0057 * pow(10,-4) * flow + \
#             -2.0706 * pow(10,-8) * pow(flow, 2)) * \
#            (1+regularization)
def var_do_predict(flow, freq):
    return {
        0: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,
                                                                                                                   2),
        1: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,
                                                                                                                   2),
        2: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,
                                                                                                                   2),
        3: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,
                                                                                                                   2),

        4: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,
                                                                                                                   2),
        5: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,
                                                                                                                   2),
        6: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,
                                                                                                                   2),
        7: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,
                                                                                                                   2),

        8: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,
                                                                                                                   2),
        9: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,
                                                                                                                   2),
        10: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(
            flow, 2),
        11: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(
            flow, 2),

        12: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(
            flow, 2),
        13: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(
            flow, 2),
        14: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(
            flow, 2),
        15: ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(
            flow, 2)}


def real_pump_press():
    """
    执行SQL查询来获取泵的实际压力数据
    :return:
    """
    bump_data = get_bumps_data_mysql()
    mean_bump_data = []
    for i in range(len(bump_data[0])):
        mean_bump_data.append(np.mean(bump_data[0][i]))  # 对输油泵数据进行均值处理,现在只有一个值

    # 筛选有功率的泵口
    real_open_bump = [[] for i in range(4)]  # 存储当前正在开启状态的泵口

    for i in range(4):
        temp = 0
        for j in range(i * 14, i * 14 + 11, 2):  # 泵压差大于0.3为开启状态
            temp = j
            if mean_bump_data[j + 1] - mean_bump_data[j] > 0.3:
                real_open_bump[i].append(mean_bump_data[j + 1] - mean_bump_data[j])
            else:
                x = 0
                real_open_bump[i].append(x)

        for k in range(2):  # 变频器频率大于20为开启状态
            if mean_bump_data[temp + 2 + k] >= 20:
                real_open_bump[i].append(mean_bump_data[temp + 2 + k])
            else:
                x = 0
                real_open_bump[i].append(x)

    real_open_bump = pd.DataFrame(real_open_bump)
    return real_open_bump


#变频泵flow--press
def predict_var_frequency_bump_pressure_v1(flow, freq, regularization, index):
    """
    计算变频泵的预测压力
    :param flow: 流量
    :param freq: 频率
    :param regularization: 正则化
    :param index: 站点索引
    :return:
    """
    return var_do_predict(flow,freq)[index] * (1+regularization[index])

# 拟合修正值
def fit_corrected_value(gpbeng_nihe_press):
    """
    利用工频实际功率，和拟合的计算功率，差值的均值作为拟合修正值
    :param gpbeng_nihe_press:
    """
    bump_data = get_bumps_data_mysql()
    mean_bump_data = []
    for i in range(len(bump_data[0])):
        mean_bump_data.append(np.mean(bump_data[0][i]))  # 对输油泵数据进行均值处理,现在只有一个值
    gpbeng_real_press = []
    for i in range(4):
        temp = 0
        for j in range(i * 14, i * 14 + 11, 2):  # 泵压差大于0.3为开启状态
            temp = j
            gpbeng_real_press.append(mean_bump_data[j + 1] - mean_bump_data[j])

    def replace_with_zero(x):
        return 0 if x < 0.3 else x

    gpbeng_real_press = [replace_with_zero(x) for x in gpbeng_real_press]

    # 创建一个 (4,6) 的 DataFrame
    gpbeng_real_press = pd.DataFrame(np.array(gpbeng_real_press).reshape(4, 6)).iloc[:, :2]
    gpbeng_real_press

    real_open = []
    for i in range(4):
        per_bump_weight = []
        for j in range(2):
            if gpbeng_real_press.iloc[i, j] > 0.3:
                # 开启状态
                per_bump_weight.append(1)
            else:
                per_bump_weight.append(0)
        real_open.append(per_bump_weight)
    real_open = pd.DataFrame(real_open)
    real_open

    gpbeng_nihe_press = pd.DataFrame([gpbeng_nihe_press[i:i + 2] for i in range(0, 8, 2)])
    # display(gpbeng_nihe_press)

    for i in range(4):
        for j in range(2):
            if real_open.iloc[i, j] == 0:
                gpbeng_nihe_press.iloc[i, j] = 0

    press_sum = 0
    num = 0
    for i in range(4):
        for j in range(2):
            if real_open.iloc[i, j] != 0:
                num += 1
                press_sum += (gpbeng_real_press.iloc[i, j] - gpbeng_nihe_press.iloc[i, j])

    press_mean = press_sum / num
    return press_mean



'''
2.95541555 -0.0010244 1.6e-07
'''
#工频泵，所有站的预测系数一样，不同点在于reg项
def predict_fixed_frequency_bump_pressure_v2(flow, regularization):
    return (2.95541555 +
            -0.0010244 * flow +
            1.6e-07 * pow(flow, 2))*(1+regularization)

def predict_var_frequency_bump_pressure_v2(flow, freq, regularization):
    """
    计算变频泵的预测压力
    :param flow: 流量
    :param freq: 频率
    :param regularization: 正则参数
    :return:
    """
    #started = 1 if (freq>math.exp(-10)) else 0
    return (((freq+0.6)**2) * 2.95541555 + \
            (freq+0.6) * -0.0010244  * flow + \
            1.6e-07 * pow(flow, 2)) * \
           (1+regularization)

def predict_fixed_frequency_bump_pressure_v3(flow,in_pressure,index):
    """
    计算工频泵的预测压力
    :param flow: 流量
    :param in_pressure: 每个站点的进站压力
    :param index: 站点索引
    :return:
    """

    return pressure_model.predict(index,flow,in_pressure)

def predict_var_frequency_bump_pressure_v3(flow,freq,in_pressure,index):
    """
    变频泵的预测压力
    :param flow: 流量
    :param freq: 频率
    :param in_pressure: 每个站点的进站压力
    :param index: 站点索引
    :return:
    """
    return pressure_model.predict_freq(index,flow,in_pressure,freq)

#获取每个站的负反馈调节系数，以及初始压力
def regularize(flow_capacity):
    """
    获取每个站的负反馈调节系数和初始压力
    :sql语句: 查询各站输油泵的压力和变频器的频率
    :param threshold:
    :return:
    """
    conn = get_conn()

    sql = """
        select a.station_name,IFNULL(b.tagc_fz08,0) as tagValue 
        from  fz_tag_station  a  left join  fz_tag_config b 
        on  a.tag_name =b.tagc_name  
        where  a.type =2 order by   a.station_id,a.sort
        """

    result = {}
    bump_data = pd.DataFrame()

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        df = pd.DataFrame(list(data)).T
        bump_data = df[1:2]
        bump_data.columns = list(df.loc['station_name'])
        bump_data.index = [0]



    #bump_data = pd.read_csv('输油泵数据.csv',header=0)
    #print(bump_data)

    # 每个站点的工频泵数量
    num_fixed_freq_bumps = 6
    # 每个站有16条泵和变频器信息
    columns = 6*2+2

    # 根据泵的数据、初始容量、泵的数量计算初始压力
    ri_zhao_reg,init_pressure = calc_reg(bump_data, flow_capacity, 0, num_fixed_freq_bumps)
    dong_hai_reg,_ = calc_reg(bump_data, flow_capacity, columns, num_fixed_freq_bumps)
    huai_an_reg,_ = calc_reg(bump_data, flow_capacity,  columns*2, num_fixed_freq_bumps)
    guan_yin_reg,_ = calc_reg(bump_data, flow_capacity, columns*3, num_fixed_freq_bumps)

    conn.close()
    print('regularize_result',{0:ri_zhao_reg,1:dong_hai_reg,2:huai_an_reg,3:guan_yin_reg},init_pressure)

    return {0:ri_zhao_reg,1:dong_hai_reg,2:huai_an_reg,3:guan_yin_reg},init_pressure



#计算修正项
#start_pos开始计算的位置
def calc_reg(bump_data, flow_capacity, start_pos, num_bumps):
    """
    函数用于计算修正项
    :param bump_data: 泵数据的DataFrame
    :param flow_capacity: 流量容量
    :param start_pos: 开始计算的位置
    :param num_bumps: 工频泵的数量
    :return:
    """
    regularzation = 0
    minimal_init_pressure = float("inf")

    # 去除无效数据
    bump_data = bump_data.dropna(axis=0)
    bump_data.loc['col_sum'] = bump_data.apply(lambda x: x.sum())
    avg_bump_data = bump_data.loc['col_sum']

    #print(avg_bump_data)

    #工频泵数据
    fixed_freq_bump_data = avg_bump_data[start_pos:start_pos+num_bumps*2]

    #工频泵阈值
    fixed_freq_threshold = 0.8
    #变频泵阈值
    var_freq_threshold = 28

    # 启动状态的工频泵提供的压力
    started_fixed_freq_bump_diff_pressure = []
    for index in np.arange(0, num_bumps * 2, 2):
        start_pressure = fixed_freq_bump_data[index]    # 入口压力
        end_pressure = fixed_freq_bump_data[index + 1]  # 出口压力
        diff_pressure = end_pressure - start_pressure   # 压力差
        if (diff_pressure > fixed_freq_threshold):      # 压力差 > 开启阈值
            if(minimal_init_pressure > start_pressure):  # 泵开启了才更新初始压力   压力一直很小就是还没开启，到开启的泵时，那个泵的入压就是初始压力值
                minimal_init_pressure = start_pressure
            started_fixed_freq_bump_diff_pressure.append(diff_pressure)   # 开启的泵的压差 存储

#   #从小到大排序
    started_fixed_freq_bump_diff_pressure.sort()
    num_started_fixed_freq_bumps = len(started_fixed_freq_bump_diff_pressure)

    #变频器数据
    var_freq_bump_data = avg_bump_data[start_pos+num_bumps*2:start_pos+num_bumps*2+2]
    num_started_var_freq_bumps = var_freq_bump_data[var_freq_bump_data > var_freq_threshold].size

    if (num_started_var_freq_bumps < num_started_fixed_freq_bumps):
        predict_pressure = do_predict_1(flow_capacity)  # 计算一个工频压力
        #python从0开始索引，因此要减一
        start_take_mean_pos = num_started_fixed_freq_bumps-num_started_var_freq_bumps-1  #
        regularzation = (np.mean(started_fixed_freq_bump_diff_pressure[start_take_mean_pos:]) - predict_pressure) / predict_pressure  # （工频泵提供压力的均值 - 预测值）/ 预测值    就是看差了多少

    if minimal_init_pressure<0.3 or minimal_init_pressure==float("inf"):
        minimal_init_pressure = 0.639

    return regularzation,minimal_init_pressure



if __name__ == "__main__":
    print(regularize(4102.0))
















