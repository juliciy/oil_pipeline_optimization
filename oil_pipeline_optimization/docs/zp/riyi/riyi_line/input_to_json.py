import pandas as pd
import numpy as np
import json
from base.mysql_conn import get_conn

'''
[
{"flow":"3661.6406","inPressure":"1.6125001","lineType":"B_LINE_0","optimizationID":0,"outPressure":"3.442578","pipeline":"0","pipelineName":"日照-东海"},
{"flow":"3762.656","inPressure":"1.19275","lineType":"B_LINE_0","optimizationID":0,"outPressure":"3.2074218","pipeline":"1","pipelineName":"东海-淮安"},
{"flow":"4418.23","inPressure":"1.618","lineType":"B_LINE_0","optimizationID":0,"outPressure":"3.704297","pipeline":"2","pipelineName":"淮安-观音"},
{"flow":"3584.7656","inPressure":"0.14825","lineType":"B_LINE_0",
"optimizationID":0,"outPressure":"2.8433595","pipeline":"3","pipelineName":"观音-仪征"}]
'''


def get_real_in_out_press(line_id=1):
    """
    获取指定线路的流量、进出站压力、线路类型
    :param line_id: 查询的线路ID
    :return:
    """
    conn = get_conn()
    sql='''
    SELECT line_type,0 as optimizationID,pipeline_id pipeline,CONCAT((select station_name 
 from fz_station_line p 
 where p.station_id = s.pre_id  
 and p.line_id = {}),'-',station_name) as pipelineName,IFNULL((select SUM(TAGC_FZ08) 
 from fz_tag_config where tagc_fz04 = pre_id 
 and pipeline_id=tagc_fz03 
 and tagc_fz07='STATION_7' 
 and tagc_fz05 = {}),0) as flow,IFNULL((select SUM(TAGC_FZ08) 
 from fz_tag_config where tagc_fz04 = station_id and pipeline_id=tagc_fz03 
 and tagc_fz07='STATION_16' and tagc_fz05 = {} ),0) as inPressure,
 IFNULL((select SUM(TAGC_FZ08)
 from fz_tag_config 
 where tagc_fz04 = pre_id and pipeline_id=tagc_fz03 and tagc_fz07='STATION_8' and tagc_fz05 = {} ),0) as outPressure 
 FROM fz_station_line as  s where sl_type != 'first' and line_id = {} order by pipeline_id
    '''.format(line_id,line_id,line_id,line_id,line_id)




    flow = []
    inPressure = []
    lineType = []
    optimizationID = []
    outPressure = []
    pipeline = []
    pipelineName = []

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        for elem in data:
            flow.append(elem['flow'])
            inPressure.append(elem['inPressure'])
            lineType.append(elem['line_type'])
            optimizationID.append(0)
            outPressure.append(elem['outPressure'])
            pipeline.append(elem['pipeline'])
            pipelineName.append(elem['pipelineName'])


    df = pd.DataFrame.from_dict({'flow': flow,
                                 'inPressure': inPressure,
                                 'lineType': lineType,
                                 'optimizationID': optimizationID,
                                 'outPressure': outPressure,
                                 'pipeline':pipeline,
                                 'pipelineName':pipelineName})


    conn.close()

    return df



def get_input_json():
    """
    获取实时进出站压力和流量数据，然后将这些数据转换为JSON格式。返回一个JSON字符串
    """
    df = get_real_in_out_press(line_id=1)

    return json.dumps(df.to_dict(orient="records"),ensure_ascii=False).replace(" ","")


