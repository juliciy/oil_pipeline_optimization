import pandas as pd
import numpy as np

from base.mysql_conn import get_conn

#
# def get_max_num_bumps():
#     settings = pd.read_csv('输油泵设定.csv',header=0)
#
#     max_num_bumps = settings.iloc[:,[1,2,3,4,5,6,7]].groupby("管段").sum().reset_index(drop=True)
#
#     max_num_bumps['max_num'] = max_num_bumps.apply(lambda x: x.sum(), axis=1)
#
#     result = max_num_bumps[['max_num']].to_dict()['max_num']
#
#     #返回如下格式{0: 5, 1: 5, 2: 4, 3: 5}
#     return result

#每个站泵的数量
def get_max_num_bumps():
    conn = get_conn()
    sql = """
        select station_id-1 as station_id, station_name,
            (select count(*) from fz_beng a 
            where a.fk_station = fz_station_line.station_id
              and a.p_type = 1
              and a.p_bengtype = 'B_TYPE_0' 
              and a.p_flag = 2  
            ) as num_bump
            from fz_station_line
            WHERE line_id = 1
            and fz_station_line.station_id < 5;
        """
    # print('get_max_num_bumps',sql)

    num_bumps = {}

    with conn.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()

        for elem in result:
            id = int(elem['station_id'])
            num = int(elem['num_bump'])
            num_bumps[id] = num

    conn.close()
    print('get_max_num_bumps_num_bumps',num_bumps)
    # 返回如下格式{0: 5, 1: 5, 2: 4, 3: 5}
    return num_bumps


#每个站开启泵的数量
def get_started_num_bumps(threshold=0.3):
    """
    包括当前日仪线各个站(不包括终点站)的泵的开启状况
    :sql语句: 查询各站输油泵的压力和变频器的频率
    :param threshold: 判断是否开启的阈值
    :return:返回一个字典，比如{0:1, 1:3, 2:2, 4:0}
    """
    conn = get_conn()

    sql = """
    select a.station_name,IFNULL(b.tagc_fz08,0) as tagValue 
    from  fz_tag_station  a  left join  fz_tag_config b 
    on  a.tag_name =b.tagc_name  
    where  a.type =2 order by   a.station_id,a.sort
    """
    # print('get_started_num_bumps',sql)
    # 记录每个站开启泵的数量
    result = {}

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        df = pd.DataFrame(list(data)).T
        # 提取的泵数据：每个站含有6个工频泵，2个变频器
        bump_data = df[1:2]
        bump_data.columns = list(df.loc['station_name'])
        bump_data.index = [0]

        #bump_data = pd.read_csv('输油泵数据.csv',header=0)
        #print(bump_data)

        # 工频泵数据，
        num_bumps = 6
        start_pos = 0

        for i in np.arange(0,4):
            started_bumps = 0

            #fixed_freq_bump_data = bump_data.iloc[:,start_pos:start_pos + num_bumps * 2]
            for j in np.arange(start_pos,start_pos + num_bumps * 2,2):


                # 入口压力 - 出口压力 > 0.3，就是开启
                delta = bump_data.iloc[0,j+1] - bump_data.iloc[0,j]
                if delta>threshold:
                    started_bumps+=1

            result[i] = started_bumps

            start_pos = start_pos + num_bumps * 2 + 2

    conn.close()
    print('get_started_num_bumps',result)
    #返回如下格式{0: 5, 1: 5, 2: 4, 3: 5}
    return result




#站点id+泵序号对应的泵点位
def get_bump_power_tag():
    return {(2,1):"ryx_DH_P2901_P",
    (2,2):"ryx_DH_P2902_P",
    (2,3):"ryx_DH_P2903_P",
    (2,4):"ryx_DH_P2904_P",
    (2,5):"ryx_DH_P2905_P",
    (2,6):"ryx_DH_P2906_P",

    (1,5):"ryx_RZZS_P1905_P",
    (1,6):"ryx_RZZS_P1906_P",
    (1,7):"ryx_RZZS_P1907_P",
    (1,8):"ryx_RZZS_P1908_P",
    (1,9):"ryx_RZZS_P1909_P",
    (1,10):"ryx_RZZS_P1910_P",

    (3,5):"ryx_HA_P3905_P",
    (3,6):"ryx_HA_P3906_P",
    (3,7):"ryx_HA_P3907_P",
    (3,8):"ryx_HA_P3908_P",
    (3,9):"ryx_HA_P3909_P",
    (3,10):"ryx_HA_P3910_P",

    (4,1):"ryx_GY_P2901_P",
    (4,2):"ryx_GY_P2902_P",
    (4,3):"ryx_GY_P2903_P",
    (4,4):"ryx_GY_P2904_P",
    (4,5):"ryx_GY_P2905_P",
    (4,6):"ryx_GY_P2906_P"}
    """
    rizhao_t = get_single_tag_value_with_time("B_RYX_RZZ_TE1111",minutes)
    donghai_t = get_single_tag_value_with_time("B_RYX_DHZ_TE2111", minutes)
    huaian_t = get_single_tag_value_with_time("B_RYX_HAZ_TE3111", minutes)
    guanyin_t = get_single_tag_value_with_time("B_RYX_GYZ_TE4111", minutes)

    return rizhao_t,donghai_t,huaian_t,guanyin_t
    """

#(站点、泵序号)对应的变频频率
def get_bump_freq_tag():
    return {(2,7):"ryx_DHZS_AI25001A",(2,8):"ryx_DHZS_AI25001A",
            (2,9):"ryx_DHZS_AI25002A",(2,10):"ryx_DHZS_AI25002A",
            (3,7):"ryx_HAZS_AI35001A",(3,8):"ryx_HAZS_AI35001A",
            (3,9):"ryx_HAZS_AI35002A",(3,10):"ryx_HAZS_AI35002A",
            (4,3):"ryx_GYZS_AI45001A",(4,4):"ryx_GYZS_AI45001A",
            (4,5):"ryx_GYZS_AI45002A",(4,6):"ryx_GYZS_AI45002A"}



#获取泵的信息，包括站id，泵id，功率点位，频率点位，入口压力点位，出口压力点位
def get_bump_tag():
    sql = """
    select id,station_id ,aa.bump_index,aa.tag_name,aa.tag_type from
(select a.station_name,
CONVERT(substring(SUBSTRING_INDEX(SUBSTRING_INDEX(a.tag_name,'PT',-1),concat(a.station_id,'9'),-1),1,2),SIGNED) as bump_index,
substring(SUBSTRING_INDEX(SUBSTRING_INDEX(a.tag_name,'PT',-1),concat(a.station_id,'9'),-1),3,1) as tag_type,
IFNULL(b.tagc_fz08,0) as tagValue ,a.tag_name ,a.station_id 
from  fz_tag_station  a  left join  fz_tag_config b 
on  a.tag_name = b.tagc_name  
where  a.type = 2
and a.tag_name  like '%PT%'
order by  a.station_id,a.sort
)aa,
(select id,fk_station ,fb.p_fz02 as bump_index from fz_beng fb 
where p_type  = 1
and p_bengtype  = 'B_TYPE_0')bb
where aa.station_id = bb.fk_station
and aa.bump_index = bb.bump_index
    """

    power_tag = get_bump_power_tag()
    freq_tag_dict = get_bump_freq_tag()

    conn = get_conn()

    bump_tag = {}
    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        for elem in data:
            bump_id = int(elem['id'])
            #A 入口 B出口
            type = elem['tag_type']
            type = "in" if type=="A" else "out"
            tag_name = elem['tag_name']
            station_id = elem['station_id']
            bump_index = elem['bump_index']

            p_tag = power_tag[(station_id,bump_index)]
            freq_tag = "" if (station_id,bump_index) not in freq_tag_dict else freq_tag_dict[(station_id,bump_index)]

            if bump_id not in bump_tag:
                bump_tag[bump_id]={}

            bump_tag[bump_id][type] = tag_name
            bump_tag[bump_id]['station_id'] = station_id
            bump_tag[bump_id]['bump_index'] = bump_index
            bump_tag[bump_id]['power'] = p_tag
            bump_tag[bump_id]['freq'] = freq_tag

    conn.close()

    return bump_tag





'''
def get_max_num_bumps_from_csv():
    settings = pd.read_csv('输油泵设定.csv',header=0)

    max_num_bumps = settings.iloc[:,[1,2,3,4,5,6,7]].groupby("管段").sum().reset_index(drop=True)

    max_num_bumps['max_num'] = max_num_bumps.apply(lambda x: x.sum(), axis=1)

    result = max_num_bumps[['max_num']].to_dict()['max_num']

    #返回如下格式{0: 5, 1: 5, 2: 4, 3: 5}
    return result




def get_started_num_bumps_from_csv():
    bump_data = pd.read_csv('输油泵数据.csv',header=0)
    result = {}

    # 工频泵数据
    num_bumps = 6
    start_pos = 0

    for i in np.arange(0,4):
        started_bumps = 0

        #fixed_freq_bump_data = bump_data.iloc[:,start_pos:start_pos + num_bumps * 2]
        for j in np.arange(start_pos,start_pos + num_bumps * 2,2):
            delta = bump_data.iloc[0,j+1] - bump_data.iloc[0,j]
            if delta>0.6:
                started_bumps+=1

        result[i] = started_bumps

        start_pos = start_pos + num_bumps * 2 + 2

    #返回如下格式{0: 5, 1: 5, 2: 4, 3: 5}
    return result

'''

if __name__ == "__main__":
    print(get_bump_tag())


