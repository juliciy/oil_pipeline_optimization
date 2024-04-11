import pandas as pd
import numpy as np
import math
from pulp import *
from base.mysql_conn import get_conn
# from riyi_line.bump_power import predict_fixed_frequency_power, predict_variable_frequency_power

'''
本部分对于各个站点输油泵的具体选择进行计算
'''
def open_bump_num():
    """
    返回dataframe，从config表中查询当前各站开泵情况，行为站，列为泵；0关，1开
    """
    conn = get_conn()
    bump_data = get_bumps_data_mysql()  # 从数据库读取泵数据
    # print('bump_data')
    # print(bump_data)
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

        # print(len(real_open_bump))
        for k in range(2):  # 变频器频率大于20为开启状态
            if mean_bump_data[temp + 2 + k] >= 20:
                real_open_bump[i].append(mean_bump_data[temp + 2 + k])
            else:
                x = 0
                real_open_bump[i].append(x)
    # real_open_bump

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
    return real_open

# 从数据库读取泵数据
def get_bumps_data_mysql():
    """
    从数据库中读取泵数据
    :return:
    """
    conn = get_conn()

    sql = """
    select a.station_name,IFNULL(b.tagc_fz08,0) as tagValue 
    from  fz_tag_station  a  left join  fz_tag_config b 
    on  a.tag_name =b.tagc_name  
    where  a.type =2 order by   a.station_id,a.sort
    """

    result = [[]]

    with conn.cursor() as cursor:
        cursor.execute(sql)
        sql_data = cursor.fetchall()
    for elem in sql_data:
        result[0].append(float(elem['tagValue']))
    conn.close()

    return result

# 从csv读取泵数据
def get_bumps_data_csv():
    """
    # 从csv读取泵数据
    :return:
    """
    path = os.path.abspath(os.path.dirname(sys.argv[0]))  # 获取文件夹绝对路径
    bump_data = pd.read_csv(path + '/输油泵数据.csv')
    bump_data = np.array(bump_data)
    bump_data = bump_data.tolist()
    return bump_data

# 数据库获取每个站泵的状态、编号
def get_bumps_open_mysql():
    """
    # 数据库获取每个站泵的状态、编号
    :return:
    """
    conn = get_conn()

    sql = """
    select b.station_id as guanduan, a.id as id , a.p_name as name , a.p_flag as flag
        FROM fz_beng a 
        left JOIN fz_station_line b 
        on a.fk_station = b.station_id
        where  a.p_type = 1  
        and a.p_bengtype ='B_TYPE_0' 
        and b.line_id = 1
        order by guanduan, id
    """

    result = [[], [], [], []]

    with conn.cursor() as cursor:
        cursor.execute(sql)
        sql_data = cursor.fetchall()
    forbidden_message = pd.DataFrame(sql_data)
    # result[0].append('rizhao')
    # result[0].append(1)
    # result[1].append('donghai')
    # result[1].append(2)
    # result[2].append('huaian')
    # result[2].append(3)
    # result[3].append('guanyin')
    # result[3].append(4)
    #
    # for elem in sql_data:
    #     guanduan = int(elem['guanduan'])
    #     flag = int(elem['flag'])
    #     result[guanduan - 1].append(flag - 1)
    # conn.close()
    #
    # return result
    return forbidden_message

# 从文件获取每个站泵的状态、编号。全部编为1不知道什么意思
# 我也不知道
def get_bumps_open_csv():
    """
    从CSV文件中读取泵的状态和编号，并将结果转换为列表返回。
    :return:
    """
    path = os.path.abspath(os.path.dirname(sys.argv[0]))  # 获取文件夹绝对路径
    bump_setting = pd.read_csv(path + '/输油泵设定.csv')
    bump_setting = np.array(bump_setting)
    bump_setting = bump_setting.tolist()
    return bump_setting



# 选泵的方法
# def bump_choose(xres, flow, in_and_out_pressure):
#     # bump_data = get_bumps_data_csv()      # 从csv读取
#     bump_data = get_bumps_data_mysql()  # 从数据库读取泵数据
#     # print('bump_data')
#     # print(bump_data)
#     mean_bump_data = []
#     for i in range(len(bump_data[0])):
#         mean_bump_data.append(np.mean(bump_data[0][i]))  # 对输油泵数据进行均值处理,现在只有一个值
#
#     # 筛选有功率的泵口
#     real_open_bump = [[] for i in range(4)]  # 存储当前正在开启状态的泵口
#     # print(real_open_bump)
#
#     for i in range(4):
#         temp = 0
#         for j in range(i * 14, i * 14 + 11, 2):  # 泵压差大于0.3为开启状态
#             temp = j
#             if mean_bump_data[j + 1] - mean_bump_data[j] > 0.3:
#                 real_open_bump[i].append(mean_bump_data[j + 1] - mean_bump_data[j])
#             else:
#                 x = 0
#                 real_open_bump[i].append(x)
#
#         print(temp)
#         for k in range(2):  # 变频器频率大于20为开启状态
#             if mean_bump_data[temp + 2 + k] >= 20:
#                 real_open_bump[i].append(mean_bump_data[temp + 2 + k])
#             else:
#                 x = 0
#                 real_open_bump[i].append(x)
#
#
#     # 计算权值
#     path = os.path.abspath(os.path.dirname(sys.argv[0]))  # 获取文件夹绝对路径
#     # bump_setting = get_bumps_open_csv()
#     bump_setting = get_bumps_open_mysql()
#     bump_choose_result = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]  # 存储最终选择的泵口
#     for i in range(4):
#         per_station_data = real_open_bump[i].copy()
#
#         per_bump_weight = []
#
#         # 计算工频泵权值
#         for j in range(6):
#             if per_station_data[j] > 0.3:
#                 #开启状态
#                 per_bump_weight.append(1)
#             else:
#                 per_bump_weight.append(0)
#
#         #优先开启的泵
#         per_bump_weight = [num * 100 + 1 for num in per_bump_weight]
#         fixed_bump_data = real_open_bump[i][:6].copy()  # 工频泵数据
#
#         for j in range(6):
#             if fixed_bump_data[j] == 0:
#                 fixed_bump_data[j] = 100
#
#         mean1 = np.mean(per_station_data[7])
#         mean2 = np.mean(per_station_data[6])
#
#         # 这块是如果两个变频器存在一个有频率
#         # 就说明原本开着一个变频泵，把这个泵的优先级提到最高
#         if (mean1 >= 20) | (mean2 >= 20):
#             min_index = fixed_bump_data.index(min(fixed_bump_data))
#             per_bump_weight[min_index] = per_bump_weight[min_index] * 2
#
#         for j in range(6):      # 一拖二变频器互斥 + 不开启泵口排除
#             if bump_setting[i][j + 2] == 0:
#                 per_bump_weight[j] = 0
#             # 日仪线3号和5号泵对应同一个变频器，要做互斥，只能开启一个。
#             # 假设3号泵原本处于开启状态，那么就把5号权值设为零
#             # 同样的道理，4号和6号也是只开启一个
#             # 权值为0的泵口就不会参与后面线性规划的计算
#             if (j == 2) | (j == 3):
#                 if per_station_data[j] > per_station_data[j + 2]:
#                     per_bump_weight[j + 2] = 0
#                     # 这个就属于零个工频一个变频的情况，这种情况是将变频泵的权值，提高到高于已经开启的工频泵
#                     if (xres[i][0] == 0) & (xres[i][1] != 0):
#                         per_bump_weight[j] += 150  # 提高变频的优先级
#                     else:
#                         per_bump_weight[j] += 50
#                 else:
#                     per_bump_weight[j] = 0
#                     if (xres[i][0] == 0) & (xres[i][1] != 0):
#                         per_bump_weight[j + 2] += 150  # 提高变频的优先级
#                     else:
#                         per_bump_weight[j + 2] += 50
#                     #per_bump_weight[j + 2] += 150  # 提高变频的优先级
#         # print("open")
#         #print(per_bump_weight)
#         # 准备开始选择输油泵
#         #变频频率
#         var_freq = xres[i][1]
#         num_bumps_to_start = xres[i][0] + math.ceil(var_freq / 500)
#         # print("num", num_bumps_to_start)
#         # print(n)
#         c = per_bump_weight  # 目标函数的系数
#         # print(c)
#         m = LpProblem("bumpchoose", LpMaximize)  # 确定求极大值
#         x1 = LpVariable("x1", 0, 1, LpInteger)
#         x2 = LpVariable("x2", 0, 1, LpInteger)
#         x3 = LpVariable("x3", 0, 1, LpInteger)
#         x4 = LpVariable("x4", 0, 1, LpInteger)
#         x5 = LpVariable("x5", 0, 1, LpInteger)
#         x6 = LpVariable("x6", 0, 1, LpInteger)  # 自变量
#         all_x = [x1, x2, x3, x4, x5, x6]
#         m += pulp.lpDot(c, all_x)
#         # 约束条件
#         m += x1 + x2 + x3 + x4 + x5 + x6 <= num_bumps_to_start
#         m.solve()
#         math_result = []
#         for j in m.variables():
#             math_result.append(j.varValue)
#         # print(math_result)
#
#         # 选择变频泵位置
#         flag = 0
#         use_index = [k for k in range(len(math_result)) if math_result[k] == 1]
#         for j in use_index:
#             if (j == 2) | (j == 3) | (j == 4) | (j == 5):
#                 flag = 1
#         use_index = use_index[::-1]
#         min_data = 1000  # 设定最小值数据，最小值坐标初值
#         min_use_index = 1000
#         if flag == 1:  # 有变频泵被选中
#             for j in use_index:
#                 if min_data > fixed_bump_data[j]:
#                     min_data = fixed_bump_data[j]
#                     min_use_index = j
#             if var_freq != 0:
#                 bump_choose_result[i][min_use_index] = var_freq  # 选择压差最小的点位赋值变频频率
#             else:
#                 bump_choose_result[i][min_use_index] = 50
#         else:  # 没有变频泵被选
#             for j in range(6):
#                 if math_result[j] == 1:
#                     bump_choose_result[i][j] = 50
#         '''
#         use_index = [k for k in range(len(math_result)) if math_result[k] == 1]
#         use_index = use_index[::-1]
#         # print(use_index)
#         min_data = 1000  # 设定最小值数据，最小值坐标初值
#         min_use_index = 1000
#         # print(per_station_data)  #ok
#         # print("up is data")
#         for j in use_index:
#             if min_data > per_station_data[j]:
#                 min_data = per_station_data[j]
#                 min_use_index = j
#
#         if var_freq != 0:
#             bump_choose_result[i][min_use_index] = var_freq  # 选择压差最小的点位赋值变频频率
#         else:
#             bump_choose_result[i][min_use_index] = 50
#         # print(min_use_index)
#         # print("up is minindex")
#         '''
#         for j in range(6):
#             if (math_result[j] == 1) & (j != min_use_index):
#                 bump_choose_result[i][j] = 50
#
#     # 按流量-功率关系计算功率
#     w = predict_fixed_frequency_power(flow)
#
#     bump_choose_output = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # 每行一个站点，前六个存储泵口选择
#     for i in range(4):
#         for j in range(6):
#             if bump_choose_result[i][j] == 50:
#                 bump_choose_output[i][j] = str(bump_choose_result[i][j]) \
#                                        + '/' + str(w)
#             else:
#                 bump_choose_output[i][j] = str(bump_choose_result[i][j]) \
#                                            + '/' + str(predict_variable_frequency_power(flow, (bump_choose_result[i][j] - 30.0)/50.0))
#         bump_choose_output[i][6] = flow
#
#
#     pressure_index = 0
#     for i in range(4):  # 进出站压力
#         bump_choose_output[i][7] = in_and_out_pressure[pressure_index]
#         pressure_index = pressure_index + 1
#         bump_choose_output[i][8] = in_and_out_pressure[pressure_index]
#         pressure_index = pressure_index + 1
#
#     # 计算功耗对比
#     sum_open_bump = 0
#     for i in range(4):
#         for j in range(6):
#             if math.ceil(real_open_bump[i][j] / 30):
#                 sum_open_bump = sum_open_bump + 1  # 计算正在工作的所有泵的数量
#     print(sum_open_bump)
#     real_power = 0
#     theory_power = 0
#     for i in range(4):
#         for j in range(6, 8):  # 筛选出变频泵，计算功率
#             if real_open_bump[i][j] > 10:
#                 if real_open_bump[i][j] > 30:
#                     real_power = real_power + predict_variable_frequency_power(flow, (real_open_bump[i][j] - 30.0) / 50.0)
#                 else:
#                     real_power = real_power + w * (real_open_bump[i][j] / 50) ** 3
#                 sum_open_bump = sum_open_bump - 1
#     real_power = real_power + sum_open_bump * w
#
#     for i in range(4):
#         for j in range(6):  # 计算按优化设想的理论功耗
#             if bump_choose_result[i][j] == 50:
#                 theory_power = theory_power + w
#             elif bump_choose_result[i][j] >= 30:
#                 theory_power = theory_power + predict_variable_frequency_power(flow, (bump_choose_result[i][j] - 30.0) / 50.0)
#             else:
#                 theory_power = theory_power + w * (bump_choose_result[i][j] / 50) ** 3
#
#     bump_choose_output[0][9] = theory_power
#     bump_choose_output[1][9] = real_power
#     if real_power == 0:
#         bump_choose_output[2][9] = 0
#     else:
#         bump_choose_output[2][9] = ((real_power - theory_power) / real_power) if real_power>0 else 1.0
#     print("real", real_power)
#     print("theory", theory_power)
#     return bump_choose_output


if __name__ == "__main__":
    xres = [[2, 41], [3, 0], [1, 41], [0, 32.9855]]
    flow = 4602.2
    pojiang = [4.9878, 1.9099, 6.3286, 2.8647, 5.3416, 1.8858, 3.9863, 0]
    print(bump_choose(xres, flow, pojiang))






