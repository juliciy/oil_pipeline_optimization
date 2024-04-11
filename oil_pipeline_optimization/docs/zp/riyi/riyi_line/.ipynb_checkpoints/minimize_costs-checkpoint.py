import time

from pyomo.environ import *
import sys
import os
from os.path import dirname
import  numpy as np
import pandas as pd


PACKAGE_DIR = dirname(dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from base.config import *
from riyi_line.bump_choose import bump_choose
from riyi_line.bump_power import predict_fixed_frequency_power,predict_variable_frequency_power
from riyi_line.bump_settings import get_max_num_bumps, get_started_num_bumps
from riyi_line.bump_pressure import regularize, \
    predict_fixed_frequency_bump_pressure_v3,predict_var_frequency_bump_pressure_v3,\
    predict_var_frequency_bump_pressure_v2,predict_fixed_frequency_bump_pressure_v2, \
    predict_var_frequency_bump_pressure_v1,predict_fixed_frequency_bump_pressure_v1
from riyi_line.output_to_json import get_json_from_records
from riyi_line.station_settings import predict_diff_pressure_between_stations, \
    get_min_inbound_pressure, get_max_outbound_pressure, \
    get_elec_price, \
    get_flow_in, get_diff_pressure_between_stations,get_last_in_pressure, get_max_inbound_pressure
from riyi_line.info_to_json import get_info_json
from riyi_line.input_to_json import get_input_json
from utils.input import write_optimize_model_result_to_mysql, write_failed_result_to_mysql, \
    write_data_problem_result_to_mysql


# bump_pressure的riyi_bump_pressure_model的fit_diff_pressure引起输出
# station_setting输出
#工频成本
def line_freq_costs(flow_capacity,elec_price):
    return predict_fixed_frequency_power(flow_capacity) * elec_price

#变频成本
def variable_freq_costs(flow_capacity,freq,elec_price):
    return predict_variable_frequency_power(flow_capacity,freq)*elec_price



def main(solver_path, flow_capacity):

    # 寻优模型
    from riyi_line.similarity import similarity_main, history_find_model_re_glo, history_find_model_el_glo
    history_find_model = similarity_main(flow_capacity)
    print("history_find_model return", history_find_model)
    from riyi_line.station_settings import glo
    riyi_success_flag = glo()
    # print("flag!!!", yichang_success_flag)
    if history_find_model == 1:
        print("use history find model")
        hitory_choice = history_find_model_re_glo()
        hitory_choice_el = history_find_model_el_glo()
        print("history choose:", hitory_choice)
        print("history choose el:", hitory_choice_el)
        write_optimize_model_result_to_mysql(2, get_input_json(), get_info_json(),
                                             get_json_from_records(hitory_choice, hitory_choice_el))
    else:
        model = ConcreteModel()
        model.I = Set(initialize=[i for i in np.arange(0, 4)])
        # model.J = Set(initialize=[i for i in np.arange(0, 4)])

        model.ElecPrice = Param(model.I, initialize=get_elec_price())
        model.Bumps = Param(model.I, default=0, initialize=get_max_num_bumps())
        model.StartedBumps = Param(model.I, default=0, initialize=get_started_num_bumps())
        model.DiffPressureLasttime = \
            Param(model.I, initialize=get_diff_pressure_between_stations(minutes=pipe_diff_pressure_minutes))

        model.LastInPressure = \
            Param(model.I, initialize=get_last_in_pressure(minutes=pipe_diff_pressure_minutes))

        reg, initial_pressure = regularize(flow_capacity)
        model.Regularzations = Param(model.I, initialize=reg)

        model.MinPressureInbound = Param(model.I, initialize=get_min_inbound_pressure())
        model.MaxPressureInbound = Param(model.I, initialize=get_max_inbound_pressure(), default=5.0)
        model.MaxPressureOutbound = Param(model.I, initialize=get_max_outbound_pressure())

        # 工频泵的开启数量
        model.x = Var(model.I, within=NonNegativeIntegers, initialize=[1, 1, 0, 0])
        # 变频泵是否开启
        model.y = Var(model.I, within=Binary, initialize=[0, 0, 0, 0])
        # 变频泵的开启频率
        model.freq = Var(model.I, within=NonNegativeReals, bounds=(0., 0.4), initialize=[0.0, 0.0, 0.0, 0.0])

        def total_costs(model):
            return sum([line_freq_costs(flow_capacity, model.ElecPrice[i]) * model.x[i] for i in model.I]) + \
                sum([variable_freq_costs(flow_capacity, model.freq[i], model.ElecPrice[i]) * model.y[i] for i in
                     model.I]) + \
                (sum((model.x[i] + model.y[i] - model.StartedBumps[i]) ** 2 for i in model.I)) * 100 + \
                (sum((model.x[i] + model.y[i]) ** 2 for i in model.I)) * 50

        model.obj = Objective(rule=total_costs, sense=minimize)

        # 输油泵开启数量限制
        def cons_bump_rule1(model, i):
            return model.x[i] + model.y[i] >= 0

        # 输油泵开启数量限制
        def cons_bump_rule2(model, i):
            return model.x[i] + model.y[i] <= model.Bumps[i]

        # 变频泵开启限制，如果没有开启，则频率为0
        def cons_bump_rule3(model, i):
            return model.freq[i] <= model.y[i]

        # model.bump_cons1 = Constraint(model.I, rule=cons_bump_rule1)
        model.bump_cons2 = Constraint(model.I, rule=cons_bump_rule2)
        model.bump_cons3 = Constraint(model.I, rule=cons_bump_rule3)

        # predict_fixed_frequency_bump_pressure_v1(flow_capacity, model.Regularzations[0])
        def predict_station0_outbound_pressure(model):
            predict_station0_outbound_pressure = predict_fixed_frequency_bump_pressure_v1(flow_capacity,
                                                                                          model.Regularzations[0], 0) * \
                                                 model.x[0] + \
                                                 predict_var_frequency_bump_pressure_v1(flow_capacity, model.freq[0],
                                                                                        model.Regularzations[0], 0) * \
                                                 model.y[0] + \
                                                 initial_pressure
            return predict_station0_outbound_pressure

        # 每个站的出站压力
        def get_station1_outbound_pressure(model):
            station0_outbound_pressure = predict_station0_outbound_pressure(model)
            station1_outbound_pressure = predict_station1_outbound_pressure(model, station0_outbound_pressure)
            return station1_outbound_pressure

        # Regularzations
        def predict_station1_outbound_pressure(model, station0_outbound_pressure):
            station1_outbound_pressure = station0_outbound_pressure - \
                                         predict_diff_pressure_between_stations(flow_capacity,
                                                                                model.DiffPressureLasttime[0], 0) + \
                                         predict_fixed_frequency_bump_pressure_v1(flow_capacity,
                                                                                  model.Regularzations[1], 1) * model.x[
                                             1] + \
                                         predict_var_frequency_bump_pressure_v1(flow_capacity,
                                                                                model.freq[1],
                                                                                model.Regularzations[1], 1) * model.y[1]
            return station1_outbound_pressure

        def get_station2_outbound_pressure(model):
            station1_outbound_pressure = get_station1_outbound_pressure(model)
            station2_outbound_pressure = predict_station2_outbound_pressure(model, station1_outbound_pressure)
            return station2_outbound_pressure

        def predict_station2_outbound_pressure(model, station1_outbound_pressure):
            station2_outbound_pressure = station1_outbound_pressure - \
                                         predict_diff_pressure_between_stations(flow_capacity,
                                                                                model.DiffPressureLasttime[1], 1) + \
                                         predict_fixed_frequency_bump_pressure_v1(flow_capacity,
                                                                                  model.Regularzations[2], 2) * model.x[
                                             2] + \
                                         predict_var_frequency_bump_pressure_v1(flow_capacity,
                                                                                model.freq[2],
                                                                                model.Regularzations[2], 2) * model.y[2]
            return station2_outbound_pressure

        def get_station3_outbound_pressure(model):
            station2_outbound_pressure = get_station2_outbound_pressure(model)
            station3_outbound_pressure = station2_outbound_pressure - \
                                         predict_diff_pressure_between_stations(flow_capacity,
                                                                                model.DiffPressureLasttime[2], 2) + \
                                         predict_fixed_frequency_bump_pressure_v1(flow_capacity,
                                                                                  model.Regularzations[3], 3) * model.x[
                                             3] + \
                                         predict_var_frequency_bump_pressure_v1(flow_capacity, model.freq[3],
                                                                                model.Regularzations[3], 3) * model.y[3]
            return station3_outbound_pressure

        # 出站压力小于最大值
        def cons_outbound_rule1(model):
            station0_outbound_pressure = predict_station0_outbound_pressure(model)
            return station0_outbound_pressure <= model.MaxPressureOutbound[0]

        # 出站压力小于最大值
        def cons_outbound_rule2(model):
            station1_outbound_pressure = get_station1_outbound_pressure(model)
            return station1_outbound_pressure <= model.MaxPressureOutbound[1]

        # 出站压力小于最大值
        def cons_outbound_rule3(model):
            station2_outbound_pressure = get_station2_outbound_pressure(model)
            return station2_outbound_pressure <= model.MaxPressureOutbound[2]

        # 出站压力小于最大值
        def cons_outbound_rule4(model):
            station3_outbound_pressure = get_station3_outbound_pressure(model)
            return station3_outbound_pressure <= model.MaxPressureOutbound[3]

        # 出站压力的下界
        def cons_outbound_rule5(model):
            station0_outbound_pressure = predict_station0_outbound_pressure(model)
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

        # 入站压力，station_0的入站不用预估
        def get_station1_inbound_pressure(model):
            station0_inbound_pressure = predict_station0_outbound_pressure(model) - \
                                        predict_diff_pressure_between_stations(flow_capacity,
                                                                               model.DiffPressureLasttime[0], 0)
            return station0_inbound_pressure

        def get_station2_inbound_pressure(model):
            station1_outbound_pressure = get_station1_outbound_pressure(model) - \
                                         predict_diff_pressure_between_stations(flow_capacity,
                                                                                model.DiffPressureLasttime[1], 1)
            return station1_outbound_pressure

        def get_station3_inbound_pressure(model):
            station2_inbound_pressure = get_station2_outbound_pressure(model) - \
                                        predict_diff_pressure_between_stations(flow_capacity,
                                                                               model.DiffPressureLasttime[2], 2)

            return station2_inbound_pressure

        def get_station4_inbound_pressure(model):
            station3_inbound_pressure = get_station3_outbound_pressure(model) - \
                                        predict_diff_pressure_between_stations(flow_capacity,
                                                                               model.DiffPressureLasttime[3], 3)

            return station3_inbound_pressure

        # 入站压力大于最小入站压力
        def cons_inbound_rule1(model):
            return get_station1_inbound_pressure(model) >= model.MinPressureInbound[0]

        def cons_inbound_rule2(model):
            return get_station2_inbound_pressure(model) >= model.MinPressureInbound[1]

        def cons_inbound_rule3(model):
            return get_station3_inbound_pressure(model) >= model.MinPressureInbound[2]

        def cons_inbound_rule4(model):
            return get_station4_inbound_pressure(model) >= model.MinPressureInbound[3]

        # 入站压力小于最大入站压力
        def cons_inbound_rule5(model):
            return get_station1_inbound_pressure(model) <= model.MaxPressureInbound[0]

        def cons_inbound_rule6(model):
            return get_station2_inbound_pressure(model) <= model.MaxPressureInbound[1]

        def cons_inbound_rule7(model):
            return get_station3_inbound_pressure(model) <= model.MaxPressureInbound[2]

        def cons_inbound_rule8(model):
            return get_station4_inbound_pressure(model) <= model.MaxPressureInbound[3]

        model.inbound_cons1 = Constraint(rule=cons_inbound_rule1)
        model.inbound_cons2 = Constraint(rule=cons_inbound_rule2)
        model.inbound_cons3 = Constraint(rule=cons_inbound_rule3)
        model.inbound_cons4 = Constraint(rule=cons_inbound_rule4)

        model.inbound_cons5 = Constraint(rule=cons_inbound_rule5)
        model.inbound_cons6 = Constraint(rule=cons_inbound_rule6)
        model.inbound_cons7 = Constraint(rule=cons_inbound_rule7)
        model.inbound_cons8 = Constraint(rule=cons_inbound_rule8)

        model.pprint()
        # model_file = model.write('model.lp')  # 输出模型文件

        # solver_path = '/home/prod/anaconda3/envs/optimize/bin/scip'
        opt = SolverFactory('scip', executable=solver_path)  # 指定求解器
        # limits/time
        # io_options = dict(add_options=["limits/time=10"])
        # model.setRealParam('limits/time', 10)
        solution = opt.solve(model, tee=True, timelimit=60)  # 调用求解器求解, tee = True 表示打印求解器的输出

        x_opt = np.array([round(value(model.x[i]), 0) for i in model.I])  # 提取最优解
        y_opt = np.array([round(value(model.y[j]), 0) for j in model.I])
        z_opt = np.array([round(value(model.freq[j]), 3) for j in model.I])

        # [1. 2. 0. 1.], [1. 1. 1. 1.], [0.4 0.4 0.4 0.4]

        print(solution.solver.termination_condition)

        obj_values = value(model.obj)
        print("optimal objective: {}".format(obj_values))
        print(x_opt, y_opt, z_opt)

        bump_status = []
        for x, y, z in zip(x_opt, y_opt, z_opt):
            bump_status.append([x, 30 + z * 50 if y == 1 else 0])

        stations_pressure = [value(predict_station0_outbound_pressure(model)),
                             value(get_station1_inbound_pressure(model)),
                             value(get_station1_outbound_pressure(model)), value(get_station2_inbound_pressure(model)),
                             value(get_station2_outbound_pressure(model)), value(get_station3_inbound_pressure(model)),
                             value(get_station3_outbound_pressure(model)), value(get_station4_inbound_pressure(model))]

        # print(bump_status)
        choice = bump_choose(bump_status, flow, stations_pressure)
        # print('************* increase ',choice[2][9])
        print(choice)

        print("optimum point: \n {},{},{} ".format(x_opt, y_opt, z_opt))
        print(bump_status)

        print('flow=', flow_capacity)

        print('---diff pressure last--')
        for i in model.I:
            print(model.DiffPressureLasttime[i])

        inbound_pressure = [stations_pressure[i] for i in np.arange(1, stations_pressure.__len__(), 2)]
        outbound_pressure = [stations_pressure[i] for i in np.arange(0, stations_pressure.__len__(), 2)]

        print('---pressure--')
        for i, o in zip(inbound_pressure, outbound_pressure):
            print(o, i, o - i)

        elec_price = [model.ElecPrice[i] for i in model.I]
        if (riyi_success_flag != 1):
            print("optimize failed as data problem")
            # write_data_problem_result_to_mysql(2, get_input_json(), get_info_json(), get_json_from_records(choice, elec_price))
        else:
                if (solution.solver.termination_condition == TerminationCondition.optimal):
                    print("optimize success")
                    write_optimize_model_result_to_mysql(2, get_input_json(), get_info_json(), get_json_from_records(choice,elec_price))
                else:
                    print("optimize failed")
                    write_failed_result_to_mysql(2, get_input_json(), get_info_json(), get_json_from_records(choice,elec_price))


    #print(value(predict_variable_frequency_bump_pressure_with_reg_v2(flow_capacity,.4,0.0)))

    '''
    choice_df = pd.DataFrame.from_records(choice)
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    choice_df.to_csv(os.path.join(path,'结果.csv'),header=False,index=False,encoding='utf-8')


    diff_pressure = [[predict_diff_pressure_between_stations(
        flow_capacity,model.DiffPressureLasttime[i],i) for i in model.I]]
    diff_pressure_df = pd.DataFrame.from_records(diff_pressure)
    diff_pressure_df.columns=['日照 - 东海段计算压差',
                              '东海 - 淮安段计算压差',
                              '淮安 - 观音段计算压差',
                                '观音 - 仪征段计算压差']
    diff_pressure_df.to_csv(os.path.join(path,'压差.csv'),
                            header=True,index=False,encoding='utf-8')
    '''





if __name__ == "__main__":
    import threading
    import sys
    start = time.time()
    solver_path = sys.argv[1]
    flow = get_flow_in(minutes=flow_minutes)        # 取指定时长流量均值，设定的2分钟
    print("flow:",flow)


    def func():
        main(solver_path,flow)

    t = threading.Thread(target=func)
    t.setDaemon(True)
    t.start()

    t.join(timeout=60 * 59)
    end = time.time()
    print("程序process_1的运行时间为：{}".format(end - start))


    import os
    import platform
    system_name = platform.system()

    if ( ("Windows" in system_name) and (os.system("tasklist | findstr /i scip.exe && taskkill /f /im scip.exe") == 0)):
        print("kill scip.exe success")
    print("riyi main thread over")














