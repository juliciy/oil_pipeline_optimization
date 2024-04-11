from base.config import max_outlet_pressure, min_inlet_pressure
from base.mysql_conn import get_conn


def get_station_settings(line_id,key):
    conn = get_conn()
    sql="""
    SELECT v.station_id as id,
    v.station_hight station_high,
    IF(v.station_out_max IS NULL or v.station_out_max = '', '0.0', v.station_out_max) station_out_max,
    v.station_low station_low,
    v.electricity_fees elec_price 
FROM fz_station_line as  s 
LEFT JOIN fz_station v on  s.station_id= v.station_id 
where s.line_id = {}
and s.sl_type in ('first','oil')
order by s.pipeline_id
    """.format(line_id)

    result={}

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()

        for elem in data:
            result[int(elem['id'])-1]=float(elem[key])

    conn.close()
    return result





def get_all_station_settings(line_id):
    """
    查询某一条线路的途径站的最小进站压力、最大进站压力、最大出站压力
    :param line_id: 石油线路id
    """
    conn = get_conn()
    sql="""
    SELECT v.station_id as id,
    v.station_hight station_high,
    IF(v.station_out_max IS NULL or v.station_out_max = '', '0.0', v.station_out_max) station_out_max,
    v.station_low station_low,
    v.electricity_fees elec_price 
FROM fz_station_line as  s 
LEFT JOIN fz_station v on  s.station_id= v.station_id 
where s.line_id = {}
and s.sl_type in ('first','oil')
order by s.pipeline_id
    """.format(line_id)

    print('get_all_station_settings',sql)

    min_inlet_p = {}
    max_inlet_p={}
    max_outlet_p = {}
    elec_price = {}

    with conn.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()
        # 对每个站的实际观测值进行范围控制，
        for elem in data:

            id = int(elem['id'])

            min_in = float(elem["station_low"])
            min_inlet_p[id] =  max(min_in, min_inlet_pressure)

            max_in = float(elem["station_high"])
            if (max_in <= 0):
                max_in = max_outlet_pressure
            else:
                max_in = min(max_in, max_outlet_pressure)
            max_inlet_p[id]= max_in


            max_out = float(elem["station_out_max"])
            if(max_out<=0):
                max_out = max_outlet_pressure
            else:
                max_out = min(max_out, max_outlet_pressure)
            max_outlet_p[id] = max_out

            elec_price[id]=float(elem["elec_price"])


    conn.close()
    return min_inlet_p,max_inlet_p,max_outlet_p,elec_price





class Line():
    def __init__(self,line_id):
        self.line_id = line_id
        self.min_inlet_p,self.max_inlet_p,self.max_outlet_p,self.elec_price =get_all_station_settings(line_id)


    def get_elec_price(self):
        return self.elec_price

    def get_min_inbound_pressure(self):
        return self.min_inlet_p

    def get_max_inbound_pressure(self):
        return self.max_inlet_p

    def get_max_outbound_pressure(self):
        return self.max_outlet_p

if __name__ == '__main__':
    test_line = Line(1)



