from pymysql import Binary

from pyomo.environ import *
import numpy as np
import pandas as pd

from riyi_line.bump_choose import get_bumps_data_mysql
from riyi_line.bump_choose import get_bumps_open_mysql, open_bump_num
from riyi_line.bump_settings import get_started_num_bumps
from riyi_line.bump_pressure import regularize, real_pump_press
from riyi_line.bump_power import real_bumpp_power
from riyi_line.output_to_json import get_json_from_records
from riyi_line.station_settings import get_max_outbound_pressure, get_elec_price, \
    get_next_min_inbound_pressure, get_next_max_inbound_pressure, \
    get_config_flow, get_diff_pressure_between_stations_config
from riyi_line.info_to_json import get_info_json
from riyi_line.input_to_json import get_input_json, get_real_in_out_press
from utils.input import write_optimize_model_result_to_mysql, write_failed_result_to_mysql
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import fsolve


# 优化计算主函数
def cal_process(flow):
    """
    主函数，接收一个参数 flow，初始化模型，定义变量、参数、目标函数和约束，然后调用求解器求解模型。
    :param flow:表示某一泵站的流量。
    :return:优化后的流量、目标函数值、泵的选择、电费价格和求解器的终止条件。
    """

    # *************************实例化模型*******************************
    model = ConcreteModel()

    # 初始化集合
    model.gongpin = Set(initialize=[0, 1, 2, 3, 4, 5, 6, 7])
    model.bianpin = Set(initialize=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    model.station = Set(initialize=[0, 1, 2, 3])
    # **********************优化变量************************************
    model.x = Var(model.gongpin, within=Binary, initialize=[0, 0, 0, 0, 0, 0, 0, 0])
    model.y = Var(model.bianpin, within=Binary, initialize=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    model.freq = Var(model.bianpin, within=NonNegativeIntegers, bounds=(0, 20),
                     initialize=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 频率变为整数，对应变频站的频率

    # 工频、变频电价参数，字典的形式，对应model.station中的电费
    elec_price = get_elec_price()
    model.gpElecPrice = Param(model.gongpin, initialize=[value for value in elec_price.values() for _ in range(2)])
    model.bpElecPrice = Param(model.bianpin, initialize=[value for value in elec_price.values() for _ in range(4)])

    # 最小进站压力限制
    model.MinPressureInbound = Param(model.station, initialize=get_next_min_inbound_pressure())  # !!! 进站要往后移一位,取数是下一站的进站限制
    # 最大进站压力限制
    model.MaxPressureInbound = Param(model.station, initialize=get_next_max_inbound_pressure())  # !!! 进站要往后移一位,取数是下一站的进站限制
    # 最大出站压力限制
    model.MaxPressureOutbound = Param(model.station, initialize=get_max_outbound_pressure())
    # 各站目前开启的泵个数
    model.StartedBumps = Param(model.station, default=0, initialize=get_started_num_bumps())
    # 各站调整系数参数 & 第0站初始压力取出
    reg, initial_pressure = regularize(flow)

    # 站间压力计算
    model.DiffPressureLasttime = Param(model.station, initialize=get_diff_pressure_between_stations_config())


    # 3、出站压力限制
    # 泵压力计算公式，再是站点压力计算公式
    # 工频  自己拟合
    def do_predict(flow):
        """
        根据流量 flow 使用预先定义的多项式公式计算每个站点的工频泵压力。
        :param flow:流量值
        :return:是一个字典，包含每个站点的预测压力
        """
        return {0: -0.000000094675 * flow ** 2 + 0.0007256603 * flow + 0.3259694955,  # 日照1
                1: -0.000000005331 * flow ** 2 + -0.0000271323 * flow + 1.9025624965,  # 日照2
                2: -0.000000019866 * flow ** 2 + 0.0000678761 * flow + 1.7679022152,  # 东海1
                3: -0.000000108118 * flow ** 2 + 0.0008077043 * flow + 0.2038966221,  # 东海2
                4: -0.000000041413 * flow ** 2 + 0.0002705021 * flow + 1.2816978385,  # 淮安1
                5: -0.000000056491 * flow ** 2 + 0.0004074756 * flow + 0.9866817504,  # 淮安2
                6: -0.000000044731 * flow ** 2 + 0.0002942832 * flow + 1.2588246840,  # 观音1
                7: -0.000000020706 * flow ** 2 + 0.0001005700 * flow + 1.6359  # 观音2
                }


    # ！！！！！！！！！！！！！！！！原始公式！！！！！！！！！！！！
    def var_do_predict(flow, freq):
        """
        用于每个站点变频泵的功率计算
        :param flow:流量
        :param freq:频率
        :return:
        """
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



    real_open_pump = real_pump_press()
    cal_open_bump = real_open_pump.copy()
    k = 0
    for i in range(4):
        for j in range(2):
            if cal_open_bump.iloc[i, j] > 0.3:
                cal_open_bump.iloc[i, j] = do_predict(flow)[k]
            k += 1

        # 变频power都填上工频，变频的再修改
    w = 0
    for i in range(4):
        for j in range(4):
            if cal_open_bump.iloc[i, j + 2] > 0.3:
                cal_open_bump.iloc[i, j + 2] = round(var_do_predict(flow, 0.4)[w], 2)  # 开启就把功率填进去
            w += 1

    # 把变频泵按频率修改press
    for i in range(4):
        freq_value = max(real_open_pump.iloc[i, 6], real_open_pump.iloc[i, 7])
        if freq_value > 0:  # 有泵开了变频

            # 找到增压最小的泵，功率改为该频率下的功率
            arr = np.array(real_open_pump.iloc[i, 2:6])
            positive_values = arr[(arr > 0)]
            min_index = np.argmin(positive_values)
            bp_index = np.where(arr == positive_values[min_index])[0][0]

            min_y_index = bp_index + 2  # 表格中的索引 # 变频泵索引取出
            # print('min_y_index',i*4+bp_index,freq_value)

            cal_open_bump.iloc[i, min_y_index] = round(var_do_predict(flow, (freq_value - 30) / 50)[i * 4 + bp_index], 2)
        else:
            # 找到增压最小的泵，功率改为该频率下的功率
            arr = np.array(real_open_pump.iloc[i, 2:6])
            positive_values = arr[(arr > 0)]
            min_index = np.argmin(positive_values)
            bp_index = np.where(arr == positive_values[min_index])[0][0]

            min_y_index = bp_index + 2  # 表格中的索引 # 变频泵索引取出
            press = real_open_pump.iloc[i, min_y_index]

            def equation(freq):
                return ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,-8) * pow(flow, 2) - press
            freq_guess = 0.0  # 初始猜测值
            freq_solution = fsolve(equation, freq_guess)
            solv_freq = freq_solution * 50 + 30
            cal_open_bump.iloc[i, min_y_index] = var_do_predict(flow, (solv_freq - 30) / 50)[i * 4 + bp_index]


    print('计算压差',cal_open_bump)
    print('实际压差',real_open_pump)


    fit_press_value = []
    for i in range(4):
        fit_press_value.append((real_open_pump.iloc[i, :6] - cal_open_bump.iloc[i, :6]).mean())

    gpReg=[value for value in fit_press_value for _ in range(2)]
    bpReg=[value for value in fit_press_value for _ in range(4)]

    model.gpReg = Param(model.gongpin, initialize=gpReg)
    model.bpReg = Param(model.bianpin, initialize=bpReg)


    # 工频泵flow--press
    def predict_fixed_frequency_bump_pressure_v1(flow, regularization, index):
        """
        用于计算工频泵的压力。
        :param flow: 流量
        :param regularization: 正则化参数
        :param index: 站点索引
        :return: 预测的压力
        """
        pressure = do_predict(flow)[index] + regularization[index]
        return pressure
    # 变频泵flow--press
    def predict_var_frequency_bump_pressure_v1(flow, freq, regularization, index):
        """
        用于计算工频泵的压力。
        :param flow: 流量
        :param freq: 变频频率
        :param regularization: 正则化参数
        :param index: 站点索引
        :return: 预测的压力
        """
        return var_do_predict(flow, freq)[index] + regularization[index]


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

    power_w = 1 / 2.65
    # 工频泵功率  power = flow * pressure * w 。其中pressure已经加入了修正值。
    def pump_gp_power(flow, regularization):
        """
        计算工频泵的功率
        :param flow: 流量
        :param regularization: 正则化参数
        :return: 每个站点的功率
        """
        return {0: (flow * power_w * ((-0.000000094675 * pow(flow, 2) + 0.0007256603 * flow + 0.3259694955) + regularization[0])),
                1: (flow * power_w * ((-0.000000005331 * pow(flow, 2) + -0.0000271323 * flow + 1.9025624965) + regularization[1])) ,
                2: (flow * power_w * ((-2.0706e-8 * pow(flow, 2) + 1.0057e-4 * flow + 1.6359) + regularization[2])) ,
                3: (flow * power_w * ((-2.0706e-8 * pow(flow, 2) + 1.0057e-4 * flow + 1.6359) + regularization[3])) ,
                4: (flow * power_w * ((-0.000000041413 * flow ** 2 + 0.0002705021 * flow + 1.2816978385) + regularization[4])),
                5: (flow * power_w* ((-0.000000056491 * flow ** 2 + 0.0004074756 * flow + 0.9866817504) + regularization[5])) ,
                6: (flow * power_w * ((-0.000000044731 * flow ** 2 + 0.0002942832 * flow + 1.2588246840) + regularization[6])) ,
                7: (flow * power_w * ((-0.000000020706 * flow ** 2 + 0.0001005700 * flow + 1.6359) + regularization[7]))
                }
    # 变频泵功率
    def pump_bp_power(flow, freq,regularization):
        """
        计算变频泵的功率
        :param flow: 流量
        :param freq: 变频频率
        :param regularization: 正则化参数
        :return: 每个站点的功率
        """
        return {0: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,2)+ regularization[0]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                1: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,2)+regularization[1]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                2: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,2)+regularization[2]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                3: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,2)+regularization[3]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),

                4: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,2)+regularization[4]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                5: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,2)+regularization[5]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                6: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,2)+regularization[6]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                7: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + - 2.0706 * pow(10,-8) * pow(flow, 2)+regularization[7]) *  power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),

                8: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + - 2.0706 * pow(10,-8) * pow(flow, 2)+regularization[8]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                9: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,2)+regularization[9]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                10: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,2)+regularization[10]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                11: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + -2.0706 * pow(10, -8) * pow(flow,2)+regularization[11]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),

                12: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + - 2.0706 * pow(10,-8) * pow(flow, 2)+regularization[12]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                13: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + - 2.0706 * pow(10,-8) * pow(flow, 2)+regularization[13]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                14: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + - 2.0706 * pow(10,-8) * pow(flow, 2)+regularization[14]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05),
                15: (flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow + - 2.0706 * pow(10,-8) * pow(flow, 2)+regularization[15]) * power_w)* (1 + (1-pow(10,-1000*(0.4-freq)))*0.05)
                }

    # 计算功率的修正值
    # 功率实际值
    real_pump_power = real_bumpp_power()
    # 功率拟合值,工频
    cal_open_bump_power = real_open_pump.copy()
    k = 0
    for i in range(4):
        for j in range(2):
            if cal_open_bump_power.iloc[i, j] > 0.3:
                cal_open_bump_power.iloc[i, j] = pump_gp_power(flow, gpReg)[k]
            k += 1
    # 变频power都填上工频，变频的再修改
    w = 0
    for i in range(4):
        for j in range(4):
            if cal_open_bump_power.iloc[i, j + 2] > 0.3:
                cal_open_bump_power.iloc[i, j + 2] = round(pump_bp_power(flow, 0.4,bpReg)[w], 2)  # 开启就把功率填进去
            w += 1
    # 把变频泵按频率修改press
    for i in range(4):
        freq_value = max(real_open_pump.iloc[i, 6], real_open_pump.iloc[i, 7])
        if freq_value > 0:  # 有泵开了变频
            # 找到增压最小的泵，功率改为该频率下的功率
            arr = np.array(real_open_pump.iloc[i, 2:6])
            positive_values = arr[(arr > 0)]
            min_index = np.argmin(positive_values)
            bp_index = np.where(arr == positive_values[min_index])[0][0]

            min_y_index = bp_index + 2  # 表格中的索引 # 变频泵索引取出
            # print('min_y_index',i*4+bp_index,freq_value)
            cal_open_bump_power.iloc[i, min_y_index] = round(pump_bp_power(flow, (freq_value - 30) / 50,bpReg)[i*4+bp_index], 2)
        else:

            # 找到增压最小的泵，功率改为该频率下的功率
            arr = np.array(real_open_pump.iloc[i, 2:6])
            positive_values = arr[(arr > 0)]
            if len(positive_values)== 1:
                if positive_values[0] > 1.6:
                    continue
            min_index = np.argmin(positive_values)
            bp_index = np.where(arr == positive_values[min_index])[0][0]

            min_y_index = bp_index + 2  # 表格中的索引 # 变频泵索引取出

            power = real_pump_power.iloc[i, min_y_index]

            def equation(freq):
                return flow * (((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,-8) * pow(flow, 2)) * power_w * 1.05 - power

            freq_guess = 0.0  # 初始猜测值
            freq_solution = fsolve(equation, freq_guess)

            cal_open_bump_power.iloc[i, min_y_index] = pump_bp_power(flow, freq_solution,bpReg)[i*4+bp_index]


    cal_open_bump_power=cal_open_bump_power.iloc[:,:6]
    diff_power_percentage = real_pump_power /cal_open_bump_power
    diff_power_percentage=diff_power_percentage.fillna(0)

    print('计算功率', cal_open_bump_power)
    print('实际功率',real_pump_power)




    # 计算每站泵功率差异百分比
    fit_power_value = []
    for i in range(4):
        power_sum = 0
        power_num = 0
        for j in range(6):
            if diff_power_percentage.iloc[i,j]!=0:
                power_sum += diff_power_percentage.iloc[i,j]
                power_num +=1
        if power_num!=0:
            fit_power_value.append((power_sum/power_num))
        else:
            fit_power_value.append(0)

    print('各站power修正百分比',fit_power_value)
    # 工频、变频功率power的调整值
    power_gpReg=[(value) for value in fit_power_value for _ in range(2)]
    power_bpReg=[(value) for value in fit_power_value for _ in range(4)]


    # 工频泵成泵 = power * elec_price
    def pump_gp_cost(flow, elec_price,regularization,power_gpReg):
        """
        计算工频泵成本
        :param flow: 流量
        :param elec_price: 电费
        :param regularization: 正则参数
        :param power_gpReg: 功率调整值
        :return: 工频泵的成本
        """
        return {0: (pump_gp_power(flow, regularization)[0]*power_gpReg[0]) * elec_price,
                1: (pump_gp_power(flow, regularization)[1]*power_gpReg[1]) * elec_price,
                2: (pump_gp_power(flow, regularization)[2]*power_gpReg[2]) * elec_price,
                3: (pump_gp_power(flow, regularization)[3]*power_gpReg[3]) * elec_price,
                4: (pump_gp_power(flow, regularization)[4]*power_gpReg[4]) * elec_price,
                5: (pump_gp_power(flow, regularization)[5]*power_gpReg[5]) * elec_price,
                6: (pump_gp_power(flow, regularization)[6]*power_gpReg[6]) * elec_price,
                7: (pump_gp_power(flow, regularization)[7]*power_gpReg[7])* elec_price
                }
    # 变频泵成泵 = power * elec_price
    def pump_bp_cost(flow, freq, elec_price,regularization,power_bpReg):
        """
        计算变频泵成本
        :param flow: 流量
        :param freq: 频率
        :param elec_price: 电费
        :param regularization: 正则参数
        :param power_bpReg: 功率调整值
        :return: 变频泵的成本
        """
        return {0: (pump_bp_power(flow, freq,regularization)[0] * power_bpReg[0]) * elec_price,
                1: (pump_bp_power(flow, freq, regularization)[1]*power_bpReg[1]) * elec_price,
                2: (pump_bp_power(flow, freq, regularization)[2]*power_bpReg[2]) * elec_price,
                3: (pump_bp_power(flow, freq, regularization)[3]*power_bpReg[3]) * elec_price,
                4: (pump_bp_power(flow, freq, regularization)[4]*power_bpReg[4]) * elec_price,
                5: (pump_bp_power(flow, freq, regularization)[5]*power_bpReg[5]) * elec_price,
                6: (pump_bp_power(flow, freq, regularization)[6]*power_bpReg[6]) * elec_price,
                7: (pump_bp_power(flow, freq, regularization)[7]*power_bpReg[7]) * elec_price,
                8: (pump_bp_power(flow, freq, regularization)[8]*power_bpReg[8]) * elec_price,
                9: (pump_bp_power(flow, freq, regularization)[9]*power_bpReg[9]) * elec_price,
                10: (pump_bp_power(flow, freq, regularization)[10]*power_bpReg[10]) * elec_price,
                11: (pump_bp_power(flow, freq, regularization)[11]*power_bpReg[11]) * elec_price,
                12: (pump_bp_power(flow, freq, regularization)[12]*power_bpReg[12]) * elec_price,
                13: (pump_bp_power(flow, freq, regularization)[13]*power_bpReg[13]) * elec_price,
                14: (pump_bp_power(flow, freq, regularization)[14]*power_bpReg[14]) * elec_price,
                15: (pump_bp_power(flow, freq, regularization)[15]*power_bpReg[15]) * elec_price
                }



    real_open = open_bump_num()
    model.gp_real_open = Param(model.gongpin, initialize=list(real_open.iloc[:, 0:2].values.flatten()))
    model.bp_real_open = Param(model.bianpin, initialize=list(real_open.iloc[:, 2:].values.flatten()))


    def sum_cost(model):
        """
        定义了模型的目标函数
        :param model: 模型对象
        :return: 所有站点的泵成本总和
        """
        all_cost = pump_gp_cost(flow, model.gpElecPrice[0],model.gpReg,power_gpReg)[0] * model.x[0] + pump_gp_cost(flow, model.gpElecPrice[1],model.gpReg,power_gpReg)[1] * \
                   model.x[1] + pump_bp_cost(flow, 0.02 * model.freq[0], model.bpElecPrice[0],model.bpReg,power_bpReg)[0] * model.y[0] + \
                   pump_bp_cost(flow, 0.02 * model.freq[1], model.bpElecPrice[1],model.bpReg,power_bpReg)[1] * model.y[1] + \
                   pump_bp_cost(flow, 0.02 * model.freq[2], model.bpElecPrice[2],model.bpReg,power_bpReg)[2] * model.y[2] + \
                   pump_bp_cost(flow, 0.02 * model.freq[3], model.bpElecPrice[3],model.bpReg,power_bpReg)[3] * model.y[3] + \
                   pump_gp_cost(flow, model.gpElecPrice[2],model.gpReg,power_gpReg)[2] * model.x[2] + pump_gp_cost(flow, model.gpElecPrice[3],model.gpReg,power_gpReg)[3] * \
                   model.x[3] + pump_bp_cost(flow, 0.02 * model.freq[4], model.bpElecPrice[4],model.bpReg,power_bpReg)[4] * model.y[4] + \
                   pump_bp_cost(flow, 0.02  * model.freq[5], model.bpElecPrice[5],model.bpReg,power_bpReg)[5] * model.y[5] + \
                   pump_bp_cost(flow, 0.02 * model.freq[6], model.bpElecPrice[6],model.bpReg,power_bpReg)[6] * model.y[6] + \
                   pump_bp_cost(flow, 0.02 * model.freq[7], model.bpElecPrice[7],model.bpReg,power_bpReg)[7] * model.y[7] + \
                   pump_gp_cost(flow, model.gpElecPrice[4],model.gpReg,power_gpReg)[4] * model.x[4] + pump_gp_cost(flow, model.gpElecPrice[5],model.gpReg,power_gpReg)[5] * \
                   model.x[5] + pump_bp_cost(flow, 0.02 * model.freq[8], model.bpElecPrice[8],model.bpReg,power_bpReg)[8] * model.y[8] + \
                   pump_bp_cost(flow, 0.02 * model.freq[9], model.bpElecPrice[9],model.bpReg,power_bpReg)[9] * model.y[9] + \
                   pump_bp_cost(flow, 0.02 * model.freq[10], model.bpElecPrice[10],model.bpReg,power_bpReg)[10] * model.y[10] + \
                   pump_bp_cost(flow, 0.02 * model.freq[11], model.bpElecPrice[11],model.bpReg,power_bpReg)[11] * model.y[11] + \
                   pump_gp_cost(flow, model.gpElecPrice[6],model.gpReg,power_gpReg)[6] * model.x[6] + pump_gp_cost(flow, model.gpElecPrice[7],model.gpReg,power_gpReg)[7] * \
                   model.x[7] + pump_bp_cost(flow, 0.02 * model.freq[12], model.bpElecPrice[12],model.bpReg,power_bpReg)[12] * model.y[12] + \
                   pump_bp_cost(flow, 0.02  * model.freq[13], model.bpElecPrice[13],model.bpReg,power_bpReg)[13] * model.y[13] + \
                   pump_bp_cost(flow, 0.02 * model.freq[14], model.bpElecPrice[14],model.bpReg,power_bpReg)[14] * model.y[14] + \
                   pump_bp_cost(flow, 0.02 * model.freq[15], model.bpElecPrice[15],model.bpReg,power_bpReg)[15] * model.y[15] + \
                   sum(abs(model.x[i] - model.gp_real_open[i]) * 800 for i in model.gongpin) + \
                   sum(abs(model.y[i] - model.bp_real_open[i]) * 800 for i in model.bianpin) + \
                   (sum(model.x[i] for i in model.gongpin) + (sum(model.y[i] for i in model.bianpin))) * 1000
        return all_cost


    model.obj1 = Objective(rule=sum_cost, sense=minimize)

    # **************定义约束条件*******************
    model.cons1 = Constraint(expr=model.y[0] * model.y[2] == 0)
    model.cons2 = Constraint(expr=model.y[1] * model.y[3] == 0)

    model.cons3 = Constraint(expr=model.y[4] * model.y[6] == 0)
    model.cons4 = Constraint(expr=model.y[5] * model.y[7] == 0)

    model.cons5 = Constraint(expr=model.y[8] * model.y[10] == 0)
    model.cons6 = Constraint(expr=model.y[9] * model.y[11] == 0)

    model.cons7 = Constraint(expr=model.y[12] * model.y[14] == 0)
    model.cons8 = Constraint(expr=model.y[13] * model.y[15] == 0)

    # 4个变频只能开一个变频
    model.cons9 = Constraint(expr=(model.y[0] * 0.02 * model.freq[0] + model.y[2] * 0.02 * model.freq[2]) * (
            model.y[1] * 0.02 * model.freq[1] + model.y[3] * 0.02 * model.freq[3]) * (0.4 - (
            model.y[0] * 0.02 * model.freq[0] + model.y[2] * 0.02 * model.freq[2])) * (0.4 - (
            model.y[1] * 0.02 * model.freq[1] + model.y[3] * 0.02 * model.freq[3])) == 0)
    model.cons10 = Constraint(expr=(model.y[4] * 0.02 * model.freq[4] + model.y[6] * 0.02 * model.freq[6]) * (
            model.y[5] * 0.02 * model.freq[5] + model.y[7] * 0.02 * model.freq[7]) * (0.4 - (
            model.y[4] * 0.02 * model.freq[4] + model.y[6] * 0.02 * model.freq[6])) * (0.4 - (
            model.y[5] * 0.02 * model.freq[5] + model.y[7] * 0.02 * model.freq[7])) == 0)
    model.cons11 = Constraint(expr=(model.y[8] * 0.02 * model.freq[8] + model.y[10] * 0.02 * model.freq[10]) * (
            model.y[9] * 0.02 * model.freq[9] + model.y[11] * 0.02 * model.freq[11]) * (0.4 - (
            model.y[8] * 0.02 * model.freq[8] + model.y[10] * 0.02 * model.freq[10])) * (0.4 - (
            model.y[9] * 0.02 * model.freq[9] + model.y[11] * 0.02 * model.freq[11])) == 0)
    model.cons12 = Constraint(expr=(model.y[12] * 0.02 * model.freq[12] + model.y[14] * 0.02 * model.freq[14]) * (
            model.y[13] * 0.02 * model.freq[13] + model.y[15] * 0.02 * model.freq[15]) * (0.4 - (
            model.y[12] * 0.02 * model.freq[12] + model.y[14] * 0.02 * model.freq[14])) * (0.4 - (
            model.y[13] * 0.02 * model.freq[13] + model.y[15] * 0.02 * model.freq[15])) == 0)


    # 2、变频开启，频率才有值
    def cons_bump_rule3(model, i):
        return 0.02 * model.freq[i] <= model.y[i]


    model.bpbump_cons = Constraint(model.bianpin, rule=cons_bump_rule3)





    # station_flow =get_station_flow()



    # 先用上一时刻压差试试
    # def fitting_pressure_drop(every_station_flow):
    #     return {
    #             0:every_station_flow[0],
    #             1:every_station_flow[1],
    #             2:every_station_flow[2],
    #             3:every_station_flow[3]
    #     }
    def fitting_pressure_drop(out_press, flow):
        return get_diff_pressure_between_stations_config()

    def pressure_drop(out_press, flow, index):
        return fitting_pressure_drop(out_press, flow)[index]

    # def pressure_drop(every_station_flow):#？？？ by liu
    #
    #     return {0: ((every_station_flow[0]/10000)**1.75)*11.13903501715338,
    #             1: ((every_station_flow[1]/10000)**1.75)*11.426737359685761,
    #             2: ((every_station_flow[2]/10000)**1.75)*12.259498623025246,
    #             3: ((every_station_flow[3]/10000)**1.75)*14.43560033655001,
    #             }


    # 第0站出站压力计算
    def predict_station0_outbound_pressure(model):
        predict_station0_outbound_pressure = initial_pressure + \
                                             predict_fixed_frequency_bump_pressure_v1(flow, model.gpReg, 0) * model.x[0] + \
                                             predict_fixed_frequency_bump_pressure_v1(flow, model.gpReg, 1) * model.x[1] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[0], model.bpReg,
                                                                                    0) * model.y[0] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[1], model.bpReg,
                                                                                    1) * model.y[1] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[2], model.bpReg,
                                                                                    2) * model.y[2] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[3], model.bpReg,
                                                                                    3) * model.y[3]
        return predict_station0_outbound_pressure


    # 第1站出站压力计算
    def get_station1_outbound_pressure(model):
        station0_outbound_pressure = predict_station0_outbound_pressure(model)
        predict_station1_outbound_pressure = station0_outbound_pressure - \
                                             pressure_drop(predict_station0_outbound_pressure(model), flow, 0) + \
                                             predict_fixed_frequency_bump_pressure_v1(flow, model.gpReg, 2) * model.x[2] + \
                                             predict_fixed_frequency_bump_pressure_v1(flow, model.gpReg, 3) * model.x[3] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[4], model.bpReg,
                                                                                    4) * model.y[4] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[5], model.bpReg,
                                                                                    5) * model.y[5] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[6], model.bpReg,
                                                                                    6) * model.y[6] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[7], model.bpReg,
                                                                                    7) * model.y[7]
        return predict_station1_outbound_pressure


    # 第2站出站压力计算
    def get_station2_outbound_pressure(model):
        station1_outbound_pressure = get_station1_outbound_pressure(model)
        predict_station2_outbound_pressure = station1_outbound_pressure - \
                                             pressure_drop(get_station1_outbound_pressure(model), flow, 1) + \
                                             predict_fixed_frequency_bump_pressure_v1(flow, model.gpReg, 4) * model.x[4] + \
                                             predict_fixed_frequency_bump_pressure_v1(flow, model.gpReg, 5) * model.x[5] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[8], model.bpReg,
                                                                                    8) * model.y[8] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[9], model.bpReg,
                                                                                    9) * model.y[9] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[10],
                                                                                    model.bpReg, 10) * model.y[10] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[11],
                                                                                    model.bpReg, 11) * model.y[11]
        return predict_station2_outbound_pressure


    # 第3站出站压力计算
    def get_station3_outbound_pressure(model):
        station2_outbound_pressure = get_station2_outbound_pressure(model)
        predict_station3_outbound_pressure = station2_outbound_pressure - \
                                             pressure_drop(get_station1_outbound_pressure(model), flow, 2) + \
                                             predict_fixed_frequency_bump_pressure_v1(flow, model.gpReg, 6) * model.x[6] + \
                                             predict_fixed_frequency_bump_pressure_v1(flow, model.gpReg, 7) * model.x[7] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[12],
                                                                                    model.bpReg, 12) * model.y[12] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[13],
                                                                                    model.bpReg, 13) * model.y[13] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[14],
                                                                                    model.bpReg, 14) * model.y[14] + \
                                             predict_var_frequency_bump_pressure_v1(flow, 0.02 * model.freq[15],
                                                                                    model.bpReg, 15) * model.y[15]
        return predict_station3_outbound_pressure


    # 第0站出站<max
    def cons_outbound_rule1(model):
        station0_outbound_pressure = predict_station0_outbound_pressure(model)  # 出站值
        return station0_outbound_pressure <= model.MaxPressureOutbound[0]


    def cons_outbound_rule2(model):
        station1_outbound_pressure = get_station1_outbound_pressure(model)
        return station1_outbound_pressure <= model.MaxPressureOutbound[1]


    def cons_outbound_rule3(model):
        station2_outbound_pressure = get_station2_outbound_pressure(model)
        return station2_outbound_pressure <= model.MaxPressureOutbound[2]


    def cons_outbound_rule4(model):
        station3_outbound_pressure = get_station3_outbound_pressure(model)
        return station3_outbound_pressure <= model.MaxPressureOutbound[3]


    # 出站压力>0      0——3
    def cons_outbound_rule5(model):
        station0_outbound_pressure = predict_station0_outbound_pressure(model)  # ！！！！！！！！！！第一站的出站最小值为5！！！！！！！！！！！
        return station0_outbound_pressure >= 0


    def cons_outbound_rule6(model):
        station1_outbound_pressure = get_station1_outbound_pressure(model)
        return station1_outbound_pressure >= 0


    def cons_outbound_rule7(model):
        station2_outbound_pressure = get_station2_outbound_pressure(model)
        return station2_outbound_pressure >= 0


    def cons_outbound_rule8(model):
        station3_outbound_pressure = get_station3_outbound_pressure(model)
        return station3_outbound_pressure >= 0


    model.outbound_cons1 = Constraint(rule=cons_outbound_rule1)
    model.outbound_cons2 = Constraint(rule=cons_outbound_rule2)
    model.outbound_cons3 = Constraint(rule=cons_outbound_rule3)
    model.outbound_cons4 = Constraint(rule=cons_outbound_rule4)

    model.outbound_cons5 = Constraint(rule=cons_outbound_rule5)
    model.outbound_cons6 = Constraint(rule=cons_outbound_rule6)
    model.outbound_cons7 = Constraint(rule=cons_outbound_rule7)
    model.outbound_cons8 = Constraint(rule=cons_outbound_rule8)


    # 4、进站压力限制
    # 计算公式
    def get_station1_inbound_pressure(model):  # 上一站的出站压力 - 站间的压降
        station0_inbound_pressure = predict_station0_outbound_pressure(model) - pressure_drop(get_station1_outbound_pressure(model), flow, 0)
        return station0_inbound_pressure


    def get_station2_inbound_pressure(model):
        station1_outbound_pressure = get_station1_outbound_pressure(model) - pressure_drop(get_station1_outbound_pressure(model), flow, 1)
        return station1_outbound_pressure


    def get_station3_inbound_pressure(model):
        station2_inbound_pressure = get_station2_outbound_pressure(model) -pressure_drop(get_station1_outbound_pressure(model), flow, 2)
        return station2_inbound_pressure


    def get_station4_inbound_pressure(model):
        station3_inbound_pressure = get_station3_outbound_pressure(model) - pressure_drop(get_station1_outbound_pressure(model), flow, 3)
        return station3_inbound_pressure


    # 入站压力 > min
    def cons_inbound_rule1(model):
        return get_station1_inbound_pressure(model) >= model.MinPressureInbound[
            0]  # 最小进站压力限制  {0: 0.4, 1: 0.5, 2: 0.5, 3: 0.5}


    def cons_inbound_rule2(model):
        return get_station2_inbound_pressure(model) >= model.MinPressureInbound[1]


    def cons_inbound_rule3(model):
        return get_station3_inbound_pressure(model) >= model.MinPressureInbound[2]


    # def cons_inbound_rule4(model):
    #     return get_station4_inbound_pressure(model) >= model.MinPressureInbound[3]
    # ！！！！！！！！！最后一站入站压力改为实际值上下浮动0.1！！！！！！！！！！！！！
    real_in_out_press = get_real_in_out_press()


    def cons_inbound_rule4(model):
        return get_station4_inbound_pressure(model) >= round(real_in_out_press.iloc[3, 1], 2)


    # 入站压力 < max
    def cons_inbound_rule5(model):
        return get_station1_inbound_pressure(model) <= model.MaxPressureInbound[0]


    def cons_inbound_rule6(model):
        return get_station2_inbound_pressure(model) <= model.MaxPressureInbound[1]


    def cons_inbound_rule7(model):
        return get_station3_inbound_pressure(model) <= model.MaxPressureInbound[2]


    # def cons_inbound_rule8(model):
    #     return get_station4_inbound_pressure(model) <= model.MaxPressureInbound[3]
    # ！！！！！！！！！最后一站入站压力改为实际值上下浮动0.3！！！！！！！！！！！！！
    def cons_inbound_rule8(model):
        return get_station4_inbound_pressure(model) <= (round(real_in_out_press.iloc[3, 1], 1) + 0.3)


    # 进站限制写到constraint里面
    model.inbound_cons1 = Constraint(rule=cons_inbound_rule1)
    model.inbound_cons2 = Constraint(rule=cons_inbound_rule2)
    model.inbound_cons3 = Constraint(rule=cons_inbound_rule3)
    model.inbound_cons4 = Constraint(rule=cons_inbound_rule4)
    model.inbound_cons5 = Constraint(rule=cons_inbound_rule5)
    model.inbound_cons6 = Constraint(rule=cons_inbound_rule6)
    model.inbound_cons7 = Constraint(rule=cons_inbound_rule7)
    model.inbound_cons8 = Constraint(rule=cons_inbound_rule8)

    # 禁用泵不参与优化计算
    forbidden_message = get_bumps_open_mysql()
    if len(forbidden_message[forbidden_message.iloc[:, 3] == 1]) > 0:
        gp_forbidden_message = forbidden_message.iloc[[0, 1, 6, 7, 12, 13, 18, 19], 3].tolist()
        bp_forbidden_message = forbidden_message.iloc[
            [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23], 3].tolist()
        # 找工频禁用编号,有禁用的话，将变量固定为0，就是不启用
        # 1是禁用，2是启用。返回禁用的索引,例如[0,1] 就是第一个站前两个泵都禁用
        gp_indices = [index for index, value in enumerate(gp_forbidden_message) if value == 1]
        if len(gp_indices) > 0:
            for i in gp_indices:
                model.x[i].fix(0)
        # 找变频禁用编号
        bp_indices = [index for index, value in enumerate(bp_forbidden_message) if value == 1]
        if len(bp_indices) > 0:
            for i in bp_indices:
                model.y[i].fix(0)

    # *************************模型求解********************************
    opt = SolverFactory('scip', executable='C:/Program Files/SCIPOptSuite 8.0.3/bin/scip')  # 指定求解器
    solution = opt.solve(model, tee=True, timelimit=60 * 14)  # 调用求解器求解, tee = True 表示打印求解器的输出

    # ***********************模型结果打印******************************
    x_opt = np.array([round(value(model.x[i]), 5) for i in model.gongpin])  # 提取最优解
    y_opt = np.array([round(value(model.y[j]), 0) for j in model.bianpin])
    z_opt = np.array([round(value(0.02 * model.freq[j]), 6) for j in model.bianpin])
    obj_values = value(model.obj1)
    print('模型状态打印', solution.solver.termination_condition)
    print("目标函数为: ", obj_values)
    print('各站工频开启情况', x_opt)
    print('各站变频开启情况', y_opt)
    print('变频频率', z_opt)

    x_opt = pd.DataFrame(x_opt.reshape(4, 2))  # 工频是否开启，0 /1 变量

    # 工频泵优化功率计算
    x_opt_optimize_power = x_opt.copy()
    u = 0
    for i in range(4):
        for j in range(2):
            x_opt_optimize_power.iloc[i, j] = pump_gp_cost(flow, 1,model.gpReg,power_gpReg)[u] * x_opt.iloc[i, j]  # 功率 = 功率 * 是否开启
            u += 1
    # x_opt_optimize_power  # 优化开启的全部功率

    z_opt = pd.DataFrame(z_opt.reshape(4, 4))  # 变频是否开启，0 /1 变量

    # 变频泵功率计算
    z_opt_optimize_power = z_opt.copy()
    u = 0
    for i in range(4):
        for j in range(4):
            if z_opt.iloc[i, j] != 0:
                z_opt_optimize_power.iloc[i, j] = pump_bp_cost(flow, z_opt.iloc[i, j], 1,model.bpReg,power_bpReg)[u]  # 功率 = 功率 * 是否开启
            u += 1
    # z_opt_optimize_power

    bump_choose_output = pd.DataFrame(np.array([0] * 40).reshape(4, 10))
    # bump_choose_output

    for i in range(4):
        for j in range(2):
            bump_choose_output.iloc[i, j] = str(x_opt.iloc[i, j] * 50) + '/' + str(
                round(x_opt_optimize_power.iloc[i, j], 2))
    for i in range(4):
        for j in range(4):
            if z_opt.iloc[i, j] != 0:
                bump_choose_output.iloc[i, j + 2] = str(round(30 + z_opt.iloc[i, j] * 50, 2)) + '/' + str(
                    round(z_opt_optimize_power.iloc[i, j], 2))
            else:
                bump_choose_output.iloc[i, j + 2] = str(0.0) + '/' + str(0.0)
    # bump_choose_output

    bump_choose_output.iloc[:, 6] = flow
    # bump_choose_output

    # 计算每一站出站压力和下一站的进站压力

    # [0站出站值，1站进站值
    # 1站出站值，2站进站值
    # 2站出站值，3站进站值
    # 3站出站值，4站进站值]
    stations_pressure = [value(predict_station0_outbound_pressure(model)), value(get_station1_inbound_pressure(model)),
                         # 根据流量 和 压力差算出来的出站
                         value(get_station1_outbound_pressure(model)), value(get_station2_inbound_pressure(model)),
                         value(get_station2_outbound_pressure(model)), value(get_station3_inbound_pressure(model)),
                         value(get_station3_outbound_pressure(model)), value(get_station4_inbound_pressure(model))]

    in_and_out_pressure = stations_pressure

    pressure_index = 0
    for i in range(4):  # 出口压力，下一站入口压力加入表格
        bump_choose_output.iloc[i, 7] = in_and_out_pressure[pressure_index]
        pressure_index = pressure_index + 1
        bump_choose_output.iloc[i, 8] = in_and_out_pressure[pressure_index]
        pressure_index = pressure_index + 1
    # bump_choose_output

    y_opt = pd.DataFrame(y_opt.reshape(4, 4))

    # 实际开泵数量
    bump_data = get_bumps_data_mysql()  # 从数据库读取泵数据

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

    real_open = []
    for i in range(4):
        per_station_data = real_open_bump[i].copy()
        per_bump_weight = []
        # 计算工频泵权值
        for j in range(6):
            if per_station_data[j] > 0.3:
                # 开启状态
                per_bump_weight.append(1)
            else:
                per_bump_weight.append(0)
        real_open.append(per_bump_weight)
        # print(per_bump_weight)
    real_open = pd.DataFrame(real_open)

    real_open_bump = pd.DataFrame(real_open_bump)

    # 当变频泵没有变频数据，但是开了变频泵时，补一个变频值
    for i in range(0, 4):
        if (real_open_bump.iloc[i, 6] == 0) & (real_open_bump.iloc[i, 7] == 0):
            arr = np.array(real_open_bump.iloc[i, 2:6])  # 变频泵压力取出
            filtered_values = arr[(arr > 0) & (arr < 1.4)]
            if len(filtered_values) > 0:  # 提供压力在（0,1.4）认为开了变频，不是工频
                min_index = np.argmin(filtered_values)
                # 通过press反推freq。反推公式索引
                function_min_index = i * 4 + min_index
                # 反推
                press = real_open_bump.iloc[i, min_index + 2]


                def equation(freq):
                    return ((freq + 0.6) ** 2) * 1.6359 + (freq + 0.6) * 1.0057 * pow(10, -4) * flow - 2.0706 * pow(10,
                                                                                                                    -8) * pow(
                        flow, 2) - press


                freq_guess = 0.0  # 初始猜测值
                freq_solution = fsolve(equation, freq_guess)
                solv_freq = freq_solution * 50 + 30
                real_open_bump.iloc[i, 6] = solv_freq

    real_open_power = real_open.copy()
    # 工频power添加
    u = 0
    for i in range(4):
        for j in range(2):
            real_open_power.iloc[i, j] = round(pump_gp_cost(flow, 1,model.gpReg,power_gpReg)[u] * real_open.iloc[i, j], 2)
            u += 1
    # 变频power都填上工频，变频的再修改
    w = 0
    for i in range(4):
        for j in range(4):
            if real_open.iloc[i, j + 2] == 1:
                real_open_power.iloc[i, j + 2] = round(pump_bp_cost(flow, 0.4, 1,model.bpReg,power_bpReg)[w], 2)  # 开启就把功率填进去
            w += 1

    # 变频频率有值，则变频泵开启的最小的那个泵为变频，其余为工频开启
    for i in range(4):
        freq_value = max(real_open_bump.iloc[i, 6], real_open_bump.iloc[i, 7])
        if freq_value > 0:  # 有泵开了变频
            # 找到增压最小的泵，功率改为该频率下的功率
            arr = np.array(real_open_bump.iloc[i, 2:6])
            positive_values = arr[(arr > 0)]
            min_index = np.argmin(positive_values)
            bp_index = np.where(arr == positive_values[min_index])[0][0]

            min_y_index = bp_index + 2  # 表格中的索引 # 变频泵索引取出
            real_open_power.iloc[i, min_y_index] = round(pump_bp_cost(flow, (freq_value - 30) / 50, 1,model.bpReg,power_bpReg)[i * 4 + bp_index], 2)

    real_all_power = real_open_power.sum().sum()  # 真实开泵全部功率
    # real_all_power

    optimize_all_power = x_opt_optimize_power.sum().sum() + z_opt_optimize_power.sum().sum()  # 优化开泵功率
    # optimize_all_power

    bump_choose_output.iloc[0, 9] = optimize_all_power / flow  # 优化功耗
    bump_choose_output.iloc[1, 9] = real_all_power / flow  # 实际功耗
    bump_choose_output.iloc[2, 9] = (real_all_power - optimize_all_power) / real_all_power  # 功耗节约百分比
    # bump_choose_output


    choice = np.array(bump_choose_output).tolist()
    print("optimum point: \n {},{},{} ".format(x_opt, y_opt, z_opt))
    print('choice', choice)


    print('flow=', flow)

    print('---diff pressure last--')

    for i in model.station:
        print(model.DiffPressureLasttime[i])

    inbound_pressure = [stations_pressure[i] for i in np.arange(1, stations_pressure.__len__(), 2)]
    outbound_pressure = [stations_pressure[i] for i in np.arange(0, stations_pressure.__len__(), 2)]

    print('---pressure--')

    for i, o in zip(inbound_pressure, outbound_pressure):
        print(o, i, o - i)

    elec_price = list(get_elec_price().values())

    return flow,obj_values,choice,elec_price,solution.solver.termination_condition


# 流量上下变化5%，输出结果。共运行11次进行比较。
circulate_result = pd.DataFrame({
                                 'flow':[],
                                 'obj_values':[]
                                 })

# 获取初始流量，依据当前流量进行寻优计算
original_flow = get_config_flow().iloc[0, 0]

# 为什么要循环三次啊？
for result_i in range(3):
    flow = original_flow * (1 + (result_i - 5) / 100)
    # print('haha',flow)

    # 对当前流量寻优计算？以日仪线途径四站为例
    flow, obj_values, choice, elec_price, reult_condition = cal_process(flow)
    data_dict = {
        'flow': flow,
        'obj_values': obj_values
    }
    # circulate_result = circulate_result.append(data_dict, ignore_index=True)
    circulate_result=pd.concat([circulate_result,pd.DataFrame([data_dict])],ignore_index=True)
# 找到最优能耗，对应流量
circulate_result_min = circulate_result[circulate_result['obj_values']==circulate_result['obj_values'].min()]
flow = circulate_result_min.iloc[0,0]
flow, obj_values, choice, elec_price, reult_condition = cal_process(flow)


if (reult_condition == TerminationCondition.optimal):
    print("optimize success")
    write_optimize_model_result_to_mysql(2, get_input_json(), get_info_json(),
                                         get_json_from_records(choice, elec_price))
else:
    print("optimize failed")
    write_failed_result_to_mysql(2, get_input_json(), get_info_json(), get_json_from_records(choice, elec_price))