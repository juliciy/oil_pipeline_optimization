import numpy as np
import pandas as pd
import json
from base.mysql_conn import get_conn
from riyi_line.station_settings import get_elec_price

#泵的启停状态
def get_bump_state():
    """
    返回一个字典，键是泵的ID，值是泵的开启状态
    """
    conn = get_conn()

    sql="""
    SELECT
 id,
 p_fz01 b_name,
 p_flag state,
IF
 (
  ( SELECT TAGC_FZ08 FROM fz_tag_config WHERE tagc_fz03 = fb.id AND tagc_fz07 = 'BENG_2' ) - 
  ( SELECT TAGC_FZ08 FROM fz_tag_config WHERE tagc_fz03 = fb.id AND tagc_fz07 = 'BENG_1' ) > 0.3,
  1,0 
 ) is_open,
 fk_station as station,
 fb.p_linetype 
FROM
 fz_beng fb 
WHERE
 p_type = 1
 AND p_bengtype = 'B_TYPE_0' 
ORDER BY
 p_fz02
    """
    bump_state = {}

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        for elem in data:
            bump_state[elem['id']] = int(elem['is_open'])


    conn.close()

    return bump_state



def get_station_df():
    """
    获取站点的电费信息，包括站点ID、站点名称和电费。返回一个包含这些信息的Pandas DataFrame
    """
    conn = get_conn()

    #查不到东海站的
    sql = """select (electricity_fees * IFNULL((select SUM(TAGC_FZ08*tagc_unitnum) 
from fz_tag_config 
where tagc_fz04 = station_id and tagc_fz07='BENG_5' 
and tagc_fz05 = 1),0)) as electrovalence_sum,
station_id id,
station_name name,
electricity_fees electrovalence 
from fz_station
where station_id in (select station_id from fz_station_line where line_id = 1)"""

    electrovalence = []
    electrovalenceSum = []
    id = []
    name = []

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        for elem in data:
            electrovalence.append(elem['electrovalence'])
            electrovalenceSum.append(elem['electrovalence_sum'])
            id.append(elem['id'])
            name.append(elem['name'])

    df = pd.DataFrame.from_dict({'electrovalence': electrovalence,
                                          'electrovalenceSum': electrovalenceSum,
                                          'id': id,
                                          'name': name})

    #修正东海站的
    bump_df = get_real_bump_data()
    bump_df = bump_df[bump_df['station']==2]
    donghai_price = get_elec_price()[1]
    donghai_electrovalence_sum = sum(bump_df['power'])*donghai_price
    #print('donghai_electrovalence_sum',donghai_electrovalence_sum)
    df.loc[df['id']==2,'electrovalenceSum'] = donghai_electrovalence_sum

    conn.close()
    return df






def get_real_bump_data():
    """
    获取泵的详细信息，包括泵的ID、名称、状态、功率和所属站点。返回一个包含这些信息的Pandas DataFrame。
    :return:
    """
    conn = get_conn()

    sql = """SELECT id,p_fz01 name,p_flag state,IFNULL((select SUM(TAGC_FZ08) 
from fz_tag_config
where tagc_fz03 = fz_beng.id and tagc_fz07='BENG_5'),0) power,
fk_station station 
FROM fz_beng where p_type = 1 and p_bengtype ='B_TYPE_0' 
order by p_fz02"""

    power = []
    state = []
    id = []
    name = []
    station = []

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        for elem in data:
            power.append(elem['power'])
            state.append(elem['state'])
            id.append(elem['id'])
            name.append(elem['name'])
            station.append(elem['station'])

    bump_state = get_bump_state()
    df = pd.DataFrame.from_dict({'power': power,
                                 'state': state,
                                 'id': id,
                                 'name': name,
                                 'station':station})

    df['state'] = df['id'].apply(lambda x: bump_state[x])
    df['power'] = df['state'] * df['power']

    def adjust_power(row):
        state = row['state']
        power = row['power']
        if(state==1 and power<=0 ):
            row['power'] = 2400.0

        return row
    df = df.apply(lambda row:adjust_power(row),axis=1)


    conn.close()
    return df


def get_info_json():
    """
    获取泵和站点的数据，然后将这些数据转换为JSON格式。返回一个JSON字符串，包含泵和站点的信息。
    """
    bump_df = get_real_bump_data()
    station_df = get_station_df()

    json_result = {'pumps': bump_df.to_dict(orient="records"),
                   'stations': station_df.to_dict(orient="records")
                   }

    return json.dumps(json_result,ensure_ascii=False).replace(" ","")


if __name__ == "__main__":
    print(get_real_bump_data())

#print(get_real_bump_data())




