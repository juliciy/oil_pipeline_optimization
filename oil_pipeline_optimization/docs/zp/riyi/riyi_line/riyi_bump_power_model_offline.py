import math

import pandas as pd
import numpy as np
import os
import sys
import time
import datetime
from sklearn import linear_model, metrics
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from riyi_line.bump_settings import get_bump_tag
from utils.ConstrainedLinearRegression import ConstrainedLinearRegression

from base.mysql_conn import get_conn
from utils.evaluation import evaluate_linear_regression_rmse, evaluate_linear_regression_rsquared
from utils.input import drop_outlier, get_tag_value_diff_with_time, get_single_tag_value_with_time, \
    write_bump_model_result_to_mysql, drop_outlier_with_width
import json

base_path = os.path.abspath(os.path.dirname(sys.argv[0]))


#工频泵
def get_fixed_freq_bump():
    """
    返回一个字典，包含工频泵的ID和它们所在的站点。
    """
    #泵及其所在的站
    return {75:0,77:0,82:1,83:1,92:2,93:2,98:3,99:3}






#{75: {'out': 'ryx_RZZS_PT1905B', 'in': 'ryx_RZZS_PT1905A'}
#关联输油泵的流量和功率数据
def get_bump_flow_power_v1(power_tag, minutes=60 * 24 * 120, freq='1MIN', threshold = 0.3):
    """
    从数据库中获取指定标签的功率数据，并筛选出超过阈值的记录。返回一个包含流量和功率数据的DataFrame。
    :param power_tag: 功率标签
    :param minutes:
    :param freq:
    :param threshold:
    :return:
    """
    print(power_tag)
    power = get_single_tag_value_with_time(power_tag,minutes,freq)
    power = power.rename({"value":"power"},axis=1)
    power = power[power['power']>threshold]

    flow = get_single_tag_value_with_time("B_RYX_RZZ_FT1111",minutes,freq)
    flow = flow.rename({"value":"flow"},axis=1)

    result = flow.merge(power,on='time',how='inner')

    return result


#关联输油泵的流量和功率数据，加上了入口压力特征
def get_bump_flow_power_v2(in_tag, power_tag, freq_tag, minutes=60 * 24 * 120, freq='1MIN', threshold = 0.3):
    """
    该函数获取流量、功率和入口压力的数据，并将它们合并成一个DataFrame。
    :param in_tag: 入口压力标签
    :param power_tag: 功率标签
    :param freq_tag: 频率标签
    :param minutes:
    :param freq:
    :param threshold:
    :return:
    """
    power = get_single_tag_value_with_time(power_tag,minutes,freq)
    power = power.rename({"value":"power"},axis=1)
    power = power[power['power']>threshold]
    power = power.groupby("time").mean("power").reset_index()

    inlet = get_single_tag_value_with_time(in_tag,minutes,freq)
    inlet = inlet.rename({"value": "in"}, axis=1)
    inlet = inlet.groupby("time").mean("in").reset_index()

    flow = get_single_tag_value_with_time("B_RYX_RZZ_FT1111",minutes,freq)
    flow = flow.rename({"value":"flow"},axis=1)
    flow = flow.groupby("time").mean("flow").reset_index()


    result = flow.merge(power,on='time',how='inner')\
        .merge(inlet, on='time', how='inner')

    '''
    if(len(freq_tag)>0):
        freq_data = get_single_tag_value_with_time(freq_tag,minutes,freq)
        freq_data = freq_data.rename({"value": "freq"}, axis=1)
        freq_data = freq_data[freq_data['freq']>1.0]
        result = result.merge(freq_data,on='time',how='inner')
    else:
        result['freq'] = 50.0
    '''

    return result




def plot_train_test(coef,X_train,y_train,X_test,y_test,pipe_id,rmse,rsquared):
    import matplotlib.pyplot as plt
    import seaborn as sns

    flow = [x[0] for x in X_train]
    in_pressure = [x[2] for x in X_train]
    y_pred = [np.dot(coef,np.insert(x,0,1.0)) for x in X_train]

    data = pd.DataFrame.from_dict({'flow': flow, 'power': y_train,
                                   'pred': y_pred, 'in_pressure':in_pressure})
    sns.scatterplot(
        data=data,
        x="flow", y="power")
    sns.lineplot(
        data=data,
        x="flow", y="pred", color='g')
    plt.title("bump id {},train_size {},rmse {},rsquared {}".format(pipe_id,len(X_train),
                                                                    round(rmse,2),round(rsquared,2)))
    plt.show()

    sns.scatterplot(
        data=data,
        x="in_pressure", y="power")
    sns.lineplot(
        data=data,
        x="in_pressure", y="pred", color='g')
    plt.title("bump id {},train_size {},rmse {},rsquared {}".format(pipe_id, len(X_train),
                                                                    round(rmse, 2), round(rsquared, 2)))
    plt.show()









def fit_bump_flow_power_linear_regression_v1(bump_id, power_tag, threshold=0.3, min_size=3, freq='1MIN'):
    """
    这个函数使用线性回归模型进行拟合。返回模型的系数、评估指标和部分数据。
    :param bump_id: 接收泵ID
    :param power_tag: 功率标签
    :param threshold:
    :param min_size:
    :param freq:
    :return:
    """
    data = get_bump_flow_power_v1(power_tag, freq=freq, threshold=threshold)

    data = drop_outlier_with_width(data, "flow", width=2.5)
    if(bump_id==102):
        data['power'] = data['power']*1000
    data = drop_outlier_with_width(data, "power", width=2.5 )

    size = len(data)
    print('bump_id ', bump_id)
    print('size ', size)

    if(size<=min_size):
        return [0.0,0.0,0.0],\
               {'rmse':None,'rsquared':None},{'flow':[],'y':[]}

    # fixed_freq_bump = get_fixed_freq_bump().keys()
    # path = os.path.abspath(os.path.dirname(sys.argv[0]))
    # if(bump_id in fixed_freq_bump):
    #     file_path = os.path.join(os.path.join(path,'data'), str(bump_id)+".csv")
    #     data.to_csv(file_path,index=False)

    max_flow = max(data['flow'])
    min_flow = min(data['flow'])

    print('min_flow ',min_flow)
    print('max_flow ',max_flow)


    reg_y = data['power'].tolist()
    flow = data['flow'].tolist()  # 流量
    flow_square = [num ** 2 for num in flow]  # 流量平方


    reg_x = []
    for i in range(len(flow)):
        reg_x.append([flow[i], flow_square[i]])  # 拼接参数

    '''
    scaler_X = MinMaxScaler()  # 实例化
    scaler_X = scaler_X.fit(reg_x)  # fit，在这里本质是生成min(x)和max(x)
    scaled_X = scaler_X.transform(reg_x)  # 通过接口导出结果
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, reg_y, test_size=0.2, random_state=42)
    '''
    X_train, X_test, y_train, y_test = train_test_split(reg_x, reg_y, test_size=0.2, random_state=42)

    try:
        model = linear_model.LinearRegression()  # 最小二乘法回归
        model.fit(X_train, y_train)

        coef = model.coef_
        coef = np.insert(coef,0,model.intercept_)

    except:
        coef = [0.0,0.0,0.0]


    rmse = evaluate_linear_regression_rmse(X_test, coef, y_test)
    rsquared = evaluate_linear_regression_rsquared(X_train, coef, y_train)


    return coef,{'rmse':rmse,'rsquared':rsquared,'train_size':len(X_train)},{'flow':flow[0:100],'y':reg_y[0:100]}


def fit_bump_flow_power_linear_regression_v2(bump_id, in_tag, power_tag, freq_tag, threshold=0.3, min_size=3, freq='1MIN'):
    """
    这个函数还包括了入口压力作为特征进行线性回归模型的拟合。
    """
    data = get_bump_flow_power_v2(in_tag, power_tag, freq_tag, freq=freq, threshold=threshold)

    data = drop_outlier_with_width(data, "flow", width=2.5)
    if(bump_id==102):
        data['power'] = data['power']*1000
    data = drop_outlier_with_width(data, "power", width=2.5)
    data = drop_outlier_with_width(data, "in", width=2.5)

    size = len(data)
    print('bump_id ', bump_id)
    print('size ', size)

    if (size <= min_size):
        return [0.0, 0.0, 0.0, 0.0], \
               {'rmse': None, 'rsquared': None}, {'flow':[],'in':[],'y':[]}

    #fixed_freq_bump_dict = get_fixed_freq_bump()
    bump_tag = get_bump_tag()

    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    if(bump_id in bump_tag):
        pipe_id = bump_tag[bump_id]['station_id']-1
        file_path = os.path.join(os.path.join(os.path.join(os.path.join(path,'data'),str(pipe_id)),"power"), str(bump_id)+".csv")
        data.to_csv(file_path,index=False)

    max_flow = max(data['flow'])
    min_flow = min(data['flow'])

    print('min_flow ', min_flow)
    print('max_flow ', max_flow)

    reg_y = data['power'].tolist()
    flow = data['flow'].tolist()  # 流量
    flow_square = [num ** 2 for num in flow]  # 流量平方
    in_p = data['in'].tolist()

    reg_x = []
    for i in range(len(flow)):
        reg_x.append([flow[i], flow_square[i], in_p[i]])  # 拼接参数

    '''
    scaler_X = MinMaxScaler()  # 实例化
    scaler_X = scaler_X.fit(reg_x)  # fit，在这里本质是生成min(x)和max(x)
    scaled_X = scaler_X.transform(reg_x)  # 通过接口导出结果
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, reg_y, test_size=0.2, random_state=42)
    '''
    X_train, X_test, y_train, y_test = train_test_split(reg_x, reg_y, test_size=0.2, random_state=42)

    try:
        model = linear_model.LinearRegression()  # 最小二乘法回归
        model.fit(X_train, y_train)

        coef = model.coef_
        coef = np.insert(coef, 0, model.intercept_)

    except:
        coef = [0.0, 0.0, 0.0, 0.0]

    rmse = evaluate_linear_regression_rmse(X_test, coef, y_test)
    rsquared = evaluate_linear_regression_rsquared(X_train, coef, y_train)

    #plot_train_test(coef,X_train,y_train,X_test,y_test,bump_id,rmse,rsquared)

    return coef, {'rmse': rmse,
                  'rsquared': rsquared,

train_size':len(X_train)}, {'flow':flow[0:100],
                                               'in_pressure':in_p[0:100],
                                               'y':reg_y[0:100]}





def fit_bump_flow_power_linear_regression_v3(bump_id, in_tag, power_tag, freq_tag, threshold=0.3, min_size=3, freq='1MIN'):
    data = get_bump_flow_power_v2(in_tag, power_tag, freq_tag, freq=freq, threshold=threshold)

    data = drop_outlier_with_width(data, "flow", width=2.5)
    if(bump_id==102):
        data['power'] = data['power']*1000
    data = drop_outlier_with_width(data, "power", width=2.5)
    data = drop_outlier_with_width(data, "in", width=2.5)

    size = len(data)
    print('bump_id ', bump_id)
    print('size ', size)

    if (size <= min_size):
        return [0.0, 0.0, 0.0, 0.0], \
               {'rmse': None, 'rsquared': None}, {'flow':[],'in':[],'y':[]}



    max_flow = max(data['flow'])
    min_flow = min(data['flow'])

    print('min_flow ', min_flow)
    print('max_flow ', max_flow)

    reg_y = data['power'].tolist()
    flow = data['flow'].tolist()  # 流量
    flow_square = [num ** 2 for num in flow]  # 流量平方
    in_p = data['in'].tolist()

    reg_x = []
    for i in range(len(flow)):
        reg_x.append([flow[i], flow_square[i], in_p[i]])  # 拼接参数

    '''
    scaler_X = MinMaxScaler()  # 实例化
    scaler_X = scaler_X.fit(reg_x)  # fit，在这里本质是生成min(x)和max(x)
    scaled_X = scaler_X.transform(reg_x)  # 通过接口导出结果
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, reg_y, test_size=0.2, random_state=42)
    '''
    X_train, X_test, y_train, y_test = train_test_split(reg_x, reg_y, test_size=0.2, random_state=42)

    try:
        model = RandomForestRegressor(n_estimators=4,random_state=0,
                                      min_samples_leaf=5)  # 最小二乘法回归
        model.fit(X_train, y_train)

        #coef = model.coef_
        #coef = np.insert(coef, 0, model.intercept_)

        y_pred = model.predict(X_test)
        rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rsquared = metrics.r2_score(y_test,y_pred)

        print('rmse ',rmse,' rqsuared ',rsquared)

        print('feature importances ',model.feature_importances_)

        return model.feature_importances_, {'rmse': rmse,
                                            'rsquared': rsquared,
                                            'train_size': len(X_train)}, \
                                           {'flow': flow[0:100],
                                            'in_pressure': in_p[0:100],
                                            'y': reg_y[0:100]}

    except Exception as e:
        print(str(e))





def run():
    all_bump_tag = get_bump_tag()

    model_type = "功率模型"

    for bump_id, v in all_bump_tag.items():

        fit_linear_regression_v1(bump_id, model_type, v)

        fit_linear_regression_v2(bump_id, model_type, v)

        fit_decision_tree_regressor_v2(bump_id, model_type, v)


def fit_decision_tree_regressor_v2(bump_id, model_type, v):
    model_func = "decision tree"
    coef, metrics, data = fit_bump_flow_power_linear_regression_v3(bump_id,
                                                                   v['in'], v['power'], v['freq'],
                                                                   min_size=20,
                                                                   freq='5MIN')
    coef = [str('%.8f' % c) for c in coef]
    model_param = ','.join(coef)
    print(coef, json.dumps(metrics))

    write_bump_model_result_to_mysql(bump_id, model_func, model_param, model_type, json.dumps(metrics),json.dumps(data))




def fit_linear_regression_v2(bump_id, model_type, v):
    model_func = "y = a + b*flow + c*flow*flow + d*in_pressure"
    coef, metrics, data = fit_bump_flow_power_linear_regression_v2(bump_id,
                                                                   v['in'], v['power'],v['freq'],
                                                                   min_size=5,
                                                                   freq='5MIN')
    coef = [str('%.8f' % c) for c in coef]
    model_param = ','.join(coef)
    print(coef, json.dumps(metrics))

    write_bump_model_result_to_mysql(bump_id, model_func, model_param, model_type, json.dumps(metrics),json.dumps(data))


def fit_linear_regression_v1(bump_id, model_type, v):
    model_func = "y = a + b*flow + c*flow*flow"
    coef, metrics, data = fit_bump_flow_power_linear_regression_v1(bump_id,
                                                                   v['power'],
                                                                   min_size=5,
                                                                   freq='5MIN')
    coef = [str('%.8f' % c) for c in coef]
    model_param = ','.join(coef)
    print(coef, json.dumps(metrics))

    write_bump_model_result_to_mysql(bump_id, model_func, model_param, model_type, json.dumps(metrics),json.dumps(data))





#78: {'out': 'ryx_RZZS_PT1907B',power': 'ryx_RZZS_P1907_P', 'in': 'ryx_RZZS_PT1907A'}
#get_bump_flow_diff_pressure("ryx_RZZS_PT1907A","ryx_RZZS_PT1907B",freq='1MIN').to_csv("78.csv")


if __name__ == "__main__":
    run()
