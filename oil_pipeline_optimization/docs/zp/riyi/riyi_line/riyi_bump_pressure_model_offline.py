import pandas as pd
import numpy as np
import os
import sys
import time
import datetime
from sklearn import linear_model, metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import math
from riyi_line.bump_settings import get_bump_tag
from utils.ConstrainedLinearRegression import ConstrainedLinearRegression

from base.mysql_conn import get_conn
from utils.evaluation import evaluate_linear_regression_rmse, evaluate_linear_regression_rsquared
from utils.input import drop_outlier, get_tag_value_diff_with_time, get_single_tag_value_with_time, \
    write_bump_model_result_to_mysql, drop_outlier_with_width
import json

base_path = os.path.abspath(os.path.dirname(sys.argv[0]))



def get_fixed_freq_bump():
    """
    返回一个字典，包含固定频率泵的ID和它们所在的站点。
    """
    #泵及其所在的站
    return {75:0,77:0,82:1,83:1,92:2,93:2,98:3,99:3}










#{75: {'out': 'ryx_RZZS_PT1905B', 'in': 'ryx_RZZS_PT1905A'}
def get_bump_flow_diff_pressure_v1(in_tag, out_tag, minutes=60 * 24 * 90, freq='1MIN', threshold = 0.3):
    """
    v1 版本返回的是单个时间点的数据
    :param in_tag: 入口压力差数据
    :param out_tag: 出口压力差数据
    :param minutes:
    :param freq:
    :param threshold:
    """
    diff_pressure = get_tag_value_diff_with_time(in_tag,out_tag,
                                      minutes,freq)
    diff_pressure = diff_pressure[diff_pressure['diff']>threshold]

    flow = get_single_tag_value_with_time("B_RYX_RZZ_FT1111",minutes,freq)
    flow = flow.rename({"value":"flow"},axis=1)


    result = flow.merge(diff_pressure,on='time',how='inner')

    return result



def get_bump_flow_diff_pressure_v2(in_tag, out_tag, minutes=60 * 24 * 90, freq='1MIN', threshold = 0.3):
    """
    v2 版本返回的是按时间分组的平均数据
    :param in_tag: 入口压力差数据
    :param out_tag: 出口压力差数据
    :param minutes:
    :param freq:
    :param threshold:
    """
    diff_pressure = get_tag_value_diff_with_time(in_tag,out_tag,
                                      minutes,freq)
    diff_pressure = diff_pressure[diff_pressure['diff']>threshold]
    diff_pressure = diff_pressure.groupby("time").mean("diff").reset_index()

    inlet = get_single_tag_value_with_time(in_tag,minutes,freq)
    inlet = inlet.rename({"value": "in"}, axis=1)
    inlet = inlet.groupby("time").mean("in").reset_index()

    flow = get_single_tag_value_with_time("B_RYX_RZZ_FT1111",minutes,freq)
    flow = flow.rename({"value":"flow"},axis=1)
    flow = flow.groupby("time").mean("flow").reset_index()

    result = flow.merge(diff_pressure,on='time',how='inner')\
        .merge(inlet, on='time', how='inner')

    return result





def plot_coef(bump_id,coef,size,min_train_flow,max_train_flow):
    flow = np.arange(min_train_flow, max_train_flow, 1.0)

    y = [np.dot(coef,[1.0,x,x*x]) for x in flow]
    # y2 =  [-12.438961426730472+0.00937426804850602*flow+-1.1614073931361102e-05*pow(flow,1.75) for flow in A]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title("bump_id="+str(bump_id)+" train size is "+str(size))
    plt.plot(flow, y, color='g', label='diff')
    # plt.plot(A,y2,color='r',label='1.75')
    plt.legend()
    plt.show()





def fit_bump_flow_diff_pressure_linear_regression_v1(bump_id, in_tag, out_tag, threshold=0.3, min_size=3, freq='1MIN'):
    """
    v1 版本只使用流量和压力差作为特征，而 v2 版本还增加了入口压力作为特征。
    :param bump_id:
    :param in_tag:
    :param out_tag:
    :param threshold:
    :param min_size:
    :param freq:
    :return:
    """
    data = get_bump_flow_diff_pressure_v1(in_tag, out_tag, freq=freq, threshold=threshold)

    data = drop_outlier_with_width(data, "flow", width=2.5)
    data = drop_outlier_with_width(data, "diff", width=2.5 )

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


    reg_y = data['diff'].tolist()
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

        #plot_coef(bump_id, coef, size, min_flow, max_flow)

    except:
        coef = [0.0,0.0,0.0]

    train_rmse = evaluate_linear_regression_rmse(X_train, coef, y_train)
    test_rmse = evaluate_linear_regression_rmse(X_test, coef, y_test)
    train_rsquared = evaluate_linear_regression_rsquared(X_train, coef, y_train)
    test_rsquared = evaluate_linear_regression_rsquared(X_test, coef, y_test)

    return coef,{'train_rmse':train_rmse, 'train_rsquared':train_rsquared,
                 'test_rmse':test_rmse,'test_rsquared':test_rsquared,
                 'train_size':len(X_train)},{'flow':flow[0:100],'y':reg_y[0:100]}


def fit_bump_flow_diff_pressure_linear_regression_v2(bump_id, in_tag, out_tag, threshold=0.3, min_size=3, freq='1MIN'):
    """
    函数使用随机森林回归器来拟合决策树模型。
    :param bump_id: 泵站编号
    :param in_tag: 入口压力的数据标签
    :param out_tag: 出口压力的数据标签
    :param threshold:
    :param min_size:
    :param freq:
    :return:
    """
    data = get_bump_flow_diff_pressure_v2(in_tag, out_tag, freq=freq, threshold=threshold)

    data = drop_outlier_with_width(data, "flow", width=2.5)
    data = drop_outlier_with_width(data, "diff", width=2.5)

    size = len(data)
    print('bump_id ', bump_id)
    print('size ', size)

    if (size <= min_size):
        return [0.0, 0.0, 0.0, 0.0], \
               {'rmse': None, 'rsquared': None}, {'flow':[],'in':[],'y':[]}

    df = data[['flow','diff','in']]
    df = df.rename(columns={'flow':'流量','diff':'压差','in':'入口压力'})
    print(df.to_json(orient="split",index=False,force_ascii=False))

    fixed_freq_bump_dict = get_fixed_freq_bump()
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    if(bump_id in fixed_freq_bump_dict):
        pipe_id = fixed_freq_bump_dict[bump_id]
        file_path = os.path.join(os.path.join(os.path.join(path,'data'),str(pipe_id)), str(bump_id)+".csv")
        data.to_csv(file_path,index=False)

    max_flow = max(data['flow'])
    min_flow = min(data['flow'])

    print('min_flow ', min_flow)
    print('max_flow ', max_flow)

    reg_y = data['diff'].tolist()
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

    train_rmse = evaluate_linear_regression_rmse(X_train, coef, y_train)
    test_rmse = evaluate_linear_regression_rmse(X_test, coef, y_test)
    train_rsquared = evaluate_linear_regression_rsquared(X_train, coef, y_train)
    test_rsquared = evaluate_linear_regression_rsquared(X_test, coef, y_test)


    return coef, {'train_rmse':train_rmse,'train_rsquared':train_rsquared,
                  'test_rmse':test_rmse,'test_rsquared':test_rsquared,
                  'train_size':len(X_train)}, {'flow':flow[0:100],
                                               'in_pressure':in_p[0:100],
                                               'y':reg_y[0:100]}



def fit_bump_flow_pressure_decision_tree_v1(bump_id, in_tag, power_tag, threshold=0.3, min_size=3, freq='1MIN'):
    """
    函数使用随机森林回归器来拟合决策树模型
    :param bump_id: 泵站编号
    :param in_tag: 入口压力的数据标签
    :param out_tag: 出口压力的数据标签
    :param threshold:
    :param min_size:
    :param freq:
    :return:
    """
    data = get_bump_flow_diff_pressure_v2(in_tag, power_tag, freq=freq, threshold=threshold)

    data = drop_outlier_with_width(data, "flow", width=2.5)
    data = drop_outlier_with_width(data, "diff", width=2.5)
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

    reg_y = data['diff'].tolist()
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
        test_rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
        test_rsquared = metrics.r2_score(y_test,y_pred)

        y_pred = model.predict(X_train)
        train_rmse = math.sqrt(metrics.mean_squared_error(y_train, y_pred))
        train_rsquared = metrics.r2_score(y_train,y_pred)
        #print('rmse ',test_rmse,' rqsuared ',test_rsquared)
        #print('feature importances ',model.feature_importances_)

        return model.feature_importances_, {'train_rmse': train_rmse,
                                            'train_rsquared': train_rsquared,
                                            'test_rmse': test_rmse,
                                            'test_rsquared': test_rsquared,
                                            'train_size': len(X_train)}, \
                                           {'flow': flow[0:100],
                                            'in_pressure': in_p[0:100],
                                            'y': reg_y[0:100]}

    except Exception as e:
        print(str(e))




def run():
    all_bump_tag = get_bump_tag()

    model_type = "特性模型"

    for bump_id, v in all_bump_tag.items():

        fit_linear_regression_v1(bump_id, model_type, v)

        fit_linear_regression_v2(bump_id, model_type, v)

        fit_decision_tree_v1(bump_id,model_type,v)



def fit_decision_tree_v1(bump_id, model_type, v):
    model_func = "random forest regressor"
    coef, metrics, data = fit_bump_flow_pressure_decision_tree_v1(bump_id,
                                                                  v['in'], v['out'],
                                                                  min_size=5,
                                                                  freq='1MIN')
    coef = [str('%.8f' % c) for c in coef]
    model_param = ','.join(coef)
    print(coef, json.dumps(metrics))
    write_bump_model_result_to_mysql(bump_id, model_func, model_param, model_type, json.dumps(metrics),json.dumps(data))



def fit_linear_regression_v2(bump_id, model_type, v):
    model_func = "y = a + b*flow + c*flow*flow + d*in_pressure"
    coef, metrics, data = fit_bump_flow_diff_pressure_linear_regression_v2(bump_id,
                                                                           v['in'], v['out'],
                                                                           min_size=5,
                                                                           freq='1MIN')
    coef = [str('%.8f' % c) for c in coef]
    model_param = ','.join(coef)
    print(coef, json.dumps(metrics))
    write_bump_model_result_to_mysql(bump_id, model_func, model_param, model_type, json.dumps(metrics),json.dumps(data))


def fit_linear_regression_v1(bump_id, model_type, v):
    model_func = "y = a + b*flow + c*flow*flow"
    coef, metrics, data = fit_bump_flow_diff_pressure_linear_regression_v1(bump_id,
                                                                           v['in'], v['out'],
                                                                           min_size=5,
                                                                           freq='1MIN')
    coef = [str('%.8f' % c) for c in coef]
    model_param = ','.join(coef)
    print(coef, json.dumps(metrics))
    write_bump_model_result_to_mysql(bump_id, model_func, model_param, model_type, json.dumps(metrics),json.dumps(data))



#print(get_bump_tag())

#78: {'out': 'ryx_RZZS_PT1907B',power': 'ryx_RZZS_P1907_P', 'in': 'ryx_RZZS_PT1907A'}
#get_bump_flow_diff_pressure("ryx_RZZS_PT1907A","ryx_RZZS_PT1907B",freq='1MIN').to_csv("78.csv")


if __name__ == "__main__":
    run()
