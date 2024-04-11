import pandas as pd
import numpy as np
import os
import sys
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
from riyi_line.ry_forcast_di_pressure_mysql import read_data_from_mysql_for_di_pressure

'''
本部分依照水力模型对压差进行训练
'''


def train_di_pressure_csv():
    # print("train")
    path = os.path.abspath(os.path.dirname(sys.argv[0]))  # 获取文件夹绝对路径
    flow_power = pd.read_csv(path + '/流量压力.csv', encoding='utf-8')
    flow_power = np.array(flow_power)
    rz_of = flow_power[:, 1]  # 日照出站流量
    rz_op = flow_power[:, 2]  # 各站出入站压力
    dh_ip = flow_power[:, 4]
    dh_op = flow_power[:, 6]
    ha_ip = flow_power[:, 8]
    ha_op = flow_power[:, 10]
    gy_ip = flow_power[:, 12]
    gy_op = flow_power[:, 14]
    yz_ip = flow_power[:, 16]
    # 使用历史数据进行拟合，计算回归系数
    rizhao_donghai_regress = regress_di_pressure(rz_of, rz_op, dh_ip)
    donghai_huaian_regress = regress_di_pressure(rz_of, dh_op, ha_ip)
    huaian_guanyi_regress = regress_di_pressure(rz_of, ha_op, gy_ip)
    guanyi_yizheng_regress = regress_di_pressure(rz_of, gy_op, yz_ip)
    # 使用拟合系数反向验证，计算拟合误差
    '''
    rd_error = test_di_pressure(rz_of, rz_op, dh_ip, rizhao_donghai_regress)
    dh_error = test_di_pressure(rz_of, dh_op, ha_ip, donghai_huaian_regress)
    hg_error = test_di_pressure(rz_of, ha_op, gy_ip, huaian_guanyi_regress)
    gy_error = test_di_pressure(rz_of, gy_op, yz_ip, guanyi_yizheng_regress)
    '''
    # print(rd_error, dh_error, hg_error, gy_error)
    rizhao_donghai_regress = rizhao_donghai_regress.tolist()
    donghai_huaian_regress = donghai_huaian_regress.tolist()
    huaian_guanyi_regress = huaian_guanyi_regress.tolist()
    guanyi_yizheng_regress = guanyi_yizheng_regress.tolist()
    return rizhao_donghai_regress, donghai_huaian_regress, huaian_guanyi_regress, guanyi_yizheng_regress


def train_di_pressure_mysql():
    # print("train")
    rz_dh_data, dh_ha_data, ha_gy_data, gy_yz_data = read_data_from_mysql_for_di_pressure()

    rz_of = rz_dh_data[:, 1]  # 日照出站流量
    rz_op = rz_dh_data[:, 2]  # 日照出站压力
    dh_ip = rz_dh_data[:, 3]  # 东海进站压力
    dh_of = dh_ha_data[:, 1]
    dh_op = dh_ha_data[:, 2]
    ha_ip = dh_ha_data[:, 3]  # 淮安进站压力
    ha_of = ha_gy_data[:, 1]
    ha_op = ha_gy_data[:, 2]
    gy_ip = ha_gy_data[:, 3]  # 观音进站压力
    gy_of = gy_yz_data[:, 1]
    gy_op = gy_yz_data[:, 2]
    yz_ip = gy_yz_data[:, 3]  # 仪征进站压力
    # 使用历史数据进行拟合，计算回归系数
    rizhao_donghai_regress = regress_di_pressure(rz_of, rz_op, dh_ip)
    donghai_huaian_regress = regress_di_pressure(dh_of, dh_op, ha_ip)
    huaian_guanyi_regress = regress_di_pressure(ha_of, ha_op, gy_ip)
    guanyi_yizheng_regress = regress_di_pressure(gy_of, gy_op, yz_ip)

    '''
    # 使用拟合系数反向验证，计算拟合误差
    test(dh_ip, dh_of, dh_op, donghai_huaian_regress, guanyi_yizheng_regress, gy_ip, gy_of, gy_op, ha_ip, ha_of, ha_op,
         huaian_guanyi_regress, rizhao_donghai_regress, rz_of, rz_op, yz_ip)

    '''
    rizhao_donghai_regress = rizhao_donghai_regress.tolist()
    donghai_huaian_regress = donghai_huaian_regress.tolist()
    huaian_guanyi_regress = huaian_guanyi_regress.tolist()
    guanyi_yizheng_regress = guanyi_yizheng_regress.tolist()

    return rizhao_donghai_regress, donghai_huaian_regress, huaian_guanyi_regress, guanyi_yizheng_regress


def test(dh_ip, dh_of, dh_op, donghai_huaian_regress, guanyi_yizheng_regress, gy_ip, gy_of, gy_op, ha_ip, ha_of, ha_op,
         huaian_guanyi_regress, rizhao_donghai_regress, rz_of, rz_op, yz_ip):
    rd_error = test_di_pressure(rz_of, rz_op, dh_ip, rizhao_donghai_regress)
    dh_error = test_di_pressure(dh_of, dh_op, ha_ip, donghai_huaian_regress)
    hg_error = test_di_pressure(ha_of, ha_op, gy_ip, huaian_guanyi_regress)
    gy_error = test_di_pressure(gy_of, gy_op, yz_ip, guanyi_yizheng_regress)
    print(rd_error, dh_error, hg_error, gy_error)


'''
本部分使用sklearn线性回归完成拟合函数
regress = regress((pin(2:end, 1) - pot(2:end, 1)), [ones(length(fin) - 1, 1) fin(2: end, 1) fin(2: end, 1).^ 1.75 
          (pin(1: length(fin) - 1, 1)-pot(1: length(fin) - 1, 1))]);
'''


def regress_di_pressure(fin, pin, pot):
    if len(fin) == 0:
        coef = [0, 0, 0, 0]
        coef = np.array(coef)
        return coef
    y = pin - pot
    train_y = y.tolist()
    reg_y = train_y[1:]  # 获取后一个时刻的压差作为y值，所以减去第一个压差
    x2 = fin.tolist()  # 第二个参数
    x2 = x2[1:]
    x3 = [num ** 1.75 for num in x2]  # 第三个参数
    x = pin - pot
    x4 = x.tolist()  # 第四个参数
    x4 = x4[:len(x4) - 1]
    reg_X = []
    for i in range(len(x2)):
        reg_X.append([x2[i], x3[i], x4[i]])  # 拼接参数

    X_train, X_test, y_train, y_test = train_test_split(
        reg_X, reg_y, test_size=0.33, random_state=42)

    reg = linear_model.LinearRegression()  # 最小二乘法回归
    reg.fit(X_train, y_train)
    # print(reg.coef_)
    # print(reg.intercept_)
    coef = reg.coef_
    coef = np.insert(coef, 0, reg.intercept_)  # 将截距补入系数中传出
    # print(result)

    # 画出测试效果20220829
    # plot_data(X_test, y_test, coef)

    return coef  # 返回拟合后的参数




'''
本部分对拟合结果误差进行测试
'''


def test_di_pressure(flow, op, ip, nihe):
    x2 = flow.tolist()  # 第二个参数
    delete = x2.pop(0)
    x3 = [num ** 1.75 for num in x2]  # 第三个参数
    x = op - ip  # 第四个参数
    x4 = x.tolist()
    delete = x4.pop(len(x4) - 1)
    reg_x = []
    for i in range(len(x2)):
        reg_x.append([1, x2[i], x3[i], x4[i]])  # 拼接参数
    yc = np.dot(reg_x, nihe)
    # 计算拟合误差
    error = (np.mean(yc) - np.mean(x)) / np.mean(yc) * 100
    print('error:')
    print(error)
    return error


def train_out_pressure_mysql_without_ar():
    # print("train")
    rz_dh_data, dh_ha_data, ha_gy_data, gy_yz_data = read_data_from_mysql_for_di_pressure()

    rz_of = rz_dh_data[:, 1]  # 日照出站流量
    rz_op = rz_dh_data[:, 2]  # 日照出站压力
    dh_ip = rz_dh_data[:, 3]  # 东海进站压力
    dh_of = dh_ha_data[:, 1]
    dh_op = dh_ha_data[:, 2]
    ha_ip = dh_ha_data[:, 3]  # 淮安进站压力
    ha_of = ha_gy_data[:, 1]
    ha_op = ha_gy_data[:, 2]
    gy_ip = ha_gy_data[:, 3]  # 观音进站压力
    gy_of = gy_yz_data[:, 1]
    gy_op = gy_yz_data[:, 2]
    yz_ip = gy_yz_data[:, 3]  # 仪征进站压力
    # 使用历史数据进行拟合，计算回归系数
    rizhao_donghai_regress = regress_out_pressure_without_ar(rz_of, rz_op, dh_ip)
    donghai_huaian_regress = regress_out_pressure_without_ar(dh_of, dh_op, ha_ip)
    huaian_guanyi_regress = regress_out_pressure_without_ar(ha_of, ha_op, gy_ip)
    guanyi_yizheng_regress = regress_out_pressure_without_ar(gy_of, gy_op, yz_ip)

    '''
    # 使用拟合系数反向验证，计算拟合误差
    test(dh_ip, dh_of, dh_op, donghai_huaian_regress, guanyi_yizheng_regress, gy_ip, gy_of, gy_op, ha_ip, ha_of, ha_op,
         huaian_guanyi_regress, rizhao_donghai_regress, rz_of, rz_op, yz_ip)
    '''

    rizhao_donghai_regress = rizhao_donghai_regress.tolist()
    donghai_huaian_regress = donghai_huaian_regress.tolist()
    huaian_guanyi_regress = huaian_guanyi_regress.tolist()
    guanyi_yizheng_regress = guanyi_yizheng_regress.tolist()
    return rizhao_donghai_regress, donghai_huaian_regress, huaian_guanyi_regress, guanyi_yizheng_regress


def regress_out_pressure_without_ar(fin, pin, pot):
    if len(fin) == 0:
        coef = [0, 0, 0, 0]
        coef = np.array(coef)
        return coef

    y = pot
    train_y = y.tolist()
    reg_y = train_y[1:]  # 获取后一个时刻的压差作为y值，所以减去第一个压差

    x2 = fin.tolist()  # 第二个参数
    x2 = x2[1:]

    x3 = [num ** 1.75 for num in x2]  # 第三个参数
    x4 = [x for x in pin]

    # x = pin - pot
    # x4 = x.tolist()  # 第四个参数
    # x4 = x4[:len(x4) - 1]
    reg_X = []
    for i in range(len(x2)):
        reg_X.append([x2[i], x3[i], x4[i]])  # 拼接参数

    X_train, X_test, y_train, y_test = train_test_split(
        reg_X, reg_y, test_size=0.33, random_state=42)

    reg = linear_model.LinearRegression()  # 最小二乘法回归
    reg.fit(X_train, y_train)
    # print(reg.coef_)
    # print(reg.intercept_)
    coef = reg.coef_
    coef = np.insert(coef, 0, reg.intercept_)  # 将截距补入系数中传出
    # print(result)

    # print(coef)
    # 画出测试效果20220829


    return coef  # 返回拟合后的参数


class RegressDiPressure:
    def __init__(self):
        # regress = train_di_pressure_mysql()  # 训练只进行一次，从sql读
        # 带自回归项的回归系数
        self.regress = train_di_pressure_mysql()  # 训练只进行一次，从csv读
        # 不带自回归项的回归系数
        # self.regress_without_ar = train_out_pressure_mysql_without_ar()
        # self.out_pressure_regress = train_out_pressure_mysql()

    '''
    本部分利用拟合得到的系数对压差进行预测
    '''

    def forecast_di_pressure(self, flow, diff_pressure, index):
        regress_coefficient = self.regress
        x2 = flow
        x3 = flow ** 1.75
        x4 = diff_pressure
        regress_coefficient = regress_coefficient[index]   # [0.7503638066177811, -0.00028671881817425907, 2.4831145687651157e-07, 0.9806509973018369]
        x = [1, x2, x3, x4]
        forcast_di_pressure = np.dot(x, regress_coefficient)
        return forcast_di_pressure


if __name__ == "__main__":
    a = RegressDiPressure()
    print(a.forecast_di_pressure(4602.2, 3.0776, 0))
    print(a.forecast_di_pressure(4602.2, 3.4650, 1))
    print(a.forecast_di_pressure(4602.2, 3.4553, 2))
    print(a.forecast_di_pressure(4602.2, 3.9860, 3))
    print(a.forecast_di_pressure(4602.2, 3.0776, 0))






