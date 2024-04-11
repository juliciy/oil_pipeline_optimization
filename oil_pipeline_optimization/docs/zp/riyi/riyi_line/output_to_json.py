import numpy as np
import pandas as pd

from base.mysql_conn import get_conn
import json

def get_station_bumps_index():
    """
    返回一个字典，键是站点ID，值是一个包含泵ID的列表。
    """
    conn = get_conn()

    sql="""
    select fk_station,id from fz_beng t
    where t.p_type  = 1
    and t.p_bengtype  = 'B_TYPE_0'
    """

    bumps = {}

    with conn.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()

        for elem in result:
            station_id = int(elem['fk_station'])
            if station_id not in bumps:
                bumps[station_id] = []

            bumps[station_id].append(int(elem['id']))

    conn.close()
    return bumps



def get_json_from_file(file):
    """
    将读取的数据转换为JSON格式。
    :param file: 读取数据的文件路径
    """
    choice_df = pd.read_csv(file,names=['0','1','2','3','4','5','flow','outbound','inbound','costs'])

    return get_json_from_df(choice_df)


def get_json_from_records(bump_choice_result,elec_price):
    """
    将数据转换为JSON格式。
    :param choice_df: 包含泵配置结果的列表
    :param elec_price: 包含电费价格的列表
    :return:
    """
    choice_df = pd.DataFrame.from_records(
        bump_choice_result,columns=['0','1','2','3','4','5','flow','outbound','inbound','costs'])

    return get_json_from_df(choice_df,elec_price)


def get_json_from_df(choice_df,elec_price):
    """
    该函数构建了几个列表，用于存储优化ID、方案、泵配置、功率、频率、站点信息、成本等数据，最后转换为JSON格式。
    :param choice_df: 包含泵配置数据的Pandas DataFrame
    :param elec_price: 包含电费价格的列表
    :return:
    """
    station_bumps = get_station_bumps_index()
    bumpOptimizationID = []
    bumpScheme = []
    pumps = []
    power = []
    frequency = []
    stations = []
    costs = []
    stationOptimizationID = []
    stationScheme = []
    inPressure = []
    outPressure = []
    flow = []
    pipeline = []
    for row in np.arange(0, 4):
        station_id = row + 1
        pipeline.append(row)
        stations.append(station_id)
        electrovalence = 0.0

        inPressure.append(round(choice_df.iloc[row]['inbound'],4))
        outPressure.append(round(choice_df.iloc[row]['outbound'],4))
        flow.append(choice_df.iloc[row]['flow'])

        for col in np.arange(0, 6):
            bumpOptimizationID.append(0)
            bumpScheme.append(0)
            pumps.append(str(station_bumps[station_id][col]))
            info = choice_df.iloc[row, col]
            frequency.append(info.split("/")[0])
            p = float(info.split("/")[1])
            electrovalence += p*elec_price[row]
            power.append(str(p))

        costs.append(electrovalence)
        stationOptimizationID.append(0)
        stationScheme.append(0)
    pumpPutList = pd.DataFrame.from_dict({'optimizationID': bumpOptimizationID,
                                          'scheme': bumpScheme,
                                          'pumps': pumps,
                                          'power': frequency, 'frequency': power})
    stationList = pd.DataFrame.from_dict({'optimizationID': stationOptimizationID,
                                          'scheme': stationScheme,
                                          'station': stations,
                                          'electrovalence': costs})
    # {"pipeline":3,"optimizationID":0,"scheme":0,"inPressure":0,"outPressure":2.8168,"electrovalence":1184.1522,"flow":"3781.0704"}
    outPutList = pd.DataFrame.from_dict({'pipeline': pipeline,
                                         'optimizationID': stationOptimizationID,
                                         'scheme': stationScheme,
                                         'inPressure': inPressure,
                                         'outPressure': outPressure,
                                         'electrovalence': costs,
                                         'flow': flow})
    json_result = {'pumpPutList': pumpPutList.to_dict(orient="records"),
                   "stationList": stationList.to_dict(orient="records"),
                   "outPutList": outPutList.to_dict(orient="records")}
    return json.dumps(json_result).replace(" ","")


