import pandas as pd
import numpy as np
import os
import sys
import time
import datetime
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils.evaluation import evaluate_test_without_constant_item
from utils.input import drop_outlier

base_path = os.path.abspath(os.path.dirname(sys.argv[0]))



def fit_diff_pressure(pipe_id):
    """
    读取指定管道的所有压力数据文件，并将它们合并成一个DataFrame。
    然后，它对数据进行预处理，包括去除异常值，并使用线性回归模型拟合数据。最后，它返回模型的系数
    :param pipe_id: 管道的ID
    """
    pressure_data_path = os.path.join(os.path.join(os.path.join(base_path, "data"),str(pipe_id)),"pressure")
    all_file_list = os.listdir(pressure_data_path)
    # all_file_list = os.listdir(pressure_data_path)

    bump_pressure_df = pd.DataFrame()
    for file_name in all_file_list:
        bump_pressure_df = pd.concat([bump_pressure_df,
                              pd.read_csv(os.path.join(pressure_data_path, file_name),header=0)])


    #flow_power_df = flow_power_df[flow_power_df['flow']>threshold]

    print("pipeline " + str(pipe_id) + " start with samples " + str(len(bump_pressure_df)))
    print('max flow ',max(bump_pressure_df['flow']))
    print('min flow ', min(bump_pressure_df['flow']))

    bump_pressure_df = drop_outlier(bump_pressure_df,"diff")
    #print(bump_pressure_df[['flow','diff','in']].describe())

    reg_y = bump_pressure_df['diff'].tolist()
    x2 = bump_pressure_df['flow'].tolist()  # 第二个参数
    x3 = [num ** 2 for num in x2]  # 第三个参数
    x4 = bump_pressure_df['in'].tolist()

    reg_x = []
    for i in range(len(x2)):
        reg_x.append([x2[i], x3[i], x4[i]])  # 拼接参数


    #最大最小归一化
    '''
    scaler_X = MinMaxScaler() #实例化
    scaler_X = scaler_X.fit(reg_x) #fit，在这里本质是生成min(x)和max(x)
    scaled_X = scaler_X.transform(reg_x) #通过接口导出结果
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, reg_y, test_size=0.33, random_state=42)
    '''

    
    X_train, X_test, y_train, y_test = train_test_split(
        reg_x, reg_y, test_size=0.2, random_state=42)


    model = linear_model.LinearRegression()  # 最小二乘法回归
    model.fit(X_train, y_train)
    coef = model.coef_
    coef = np.insert(coef,0,model.intercept_)

    evaluate_test_without_constant_item(X_test, coef, y_test)

    return coef







class riyi_bump_pressure_model():
    def __init__(self):
        self.coefs = []

        pipe0_coef = fit_diff_pressure(0)
        self.coefs.append(pipe0_coef)

        pipe1_coef = fit_diff_pressure(1)
        self.coefs.append(pipe1_coef)

        pipe2_coef = fit_diff_pressure(2)
        self.coefs.append(pipe2_coef)

        #pipe3的训练数据比较少，用其它站的模型作投票法来得到最终预测结果
        coef = fit_diff_pressure(3)
        pipe3_coef = [0.3*w+0.3*x+0.4*y+0.0*z for w,x,y,z in zip(pipe0_coef,pipe1_coef,pipe2_coef,coef)]
        self.coefs.append(pipe3_coef)



    def predict(self, pipe_id,flow,in_p):
        input  = [1,flow,flow*flow,in_p]
        return np.dot(self.coefs[pipe_id], input)

    def predict_freq(self, pipe_id,flow,in_p,freq):
        input  = [1,flow,flow*flow,in_p]
        coef = self.coefs[pipe_id]
        return (freq+0.6)**2 * input[0]*coef[0]+ \
               (freq+0.6) * input[1]*coef[1] + \
               input[2]*coef[2] + \
               (freq+0.6)**2 * input[3]*coef[3]




if __name__ == "__main__":

    m = riyi_bump_pressure_model()


    print(m.predict(0, 3400,0.7))
    print(m.predict(1, 4100,1.5))
    print(m.predict(2, 4100,0.7))
    print(m.predict(2, 4100, 1.7))
    print(m.predict(3, 4500, 1.6))
    print(m.predict(3, 3500, 2.6))

    print(m.predict_freq(2,4100,1.7,(49-30.0)/50.0))
    print(m.predict_freq(3,4500,0.6,(30.0-30.0)/50.0))











