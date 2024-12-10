import socket
import time
from datetime import datetime

default_config = { 
    'Env':
        {'FileType' : 'text/acmi/tacview',
         'FileVersion' : '2.1',
         'ReferenceLongitude' : '36',
         'ReferenceLatitude' : '37',
         'ReferenceTime' : '2022-07-06T08:39:21.153Z',
         'DeltaTime' : '0.2'},
    'Plane':
        {'101' : {'Type':'Air+FixedWing', 'Model':'GJ-11', 'Color':'Red'},
         '102' : {'Type':'Air+FixedWing', 'Model':'F-22A', 'Color':'Blue'}},
    'RealTime':False,
    'Info':{'IP':'localhost','port':7788}
    }

class TacviewRecorder():
    def __init__(self, config):
        '''
        config: Dict
            { 
                'Env':
                    {'FileType' : 'text/acmi/tacview',
                     'FileVersion' : '2.1',
                     'ReferenceLongitude' : '36',
                     'ReferenceLatitude' : '37',
                     'ReferenceTime' : '2022-07-06T08:39:21.153Z',
                     'DeltaTime' : '0.2'},
                'Plane':
                    {'101' : {'Type':'Air+FixedWing', 'Model':'GJ-11', 'Color':'Red'},
                     '102' : {'Type':'Air+FixedWing', 'Model':'F-22A', 'Color':'Blue'}},
                'RealTime':False,
                'Info':{'IP':'localhost','port':7788}
                }
            
        '''
        self.FileType = config['Env']['FileType']
        self.FileVersion = config['Env']['FileVersion']

        self.ReferenceTime = config['Env']['ReferenceTime']
        self.dt = float(config['Env']['DeltaTime'])
        
        self.plane_dict = config['Plane']
        #heading
        self.data = 'FileType='+self.FileType+'\n' + \
                    'FileVersion='+self.FileVersion+'\n' + \
                    '0,ReferenceTime='+self.ReferenceTime+'\n'
                    
        self.RealTime = config.get('RealTime',False)
        print(f'realtime render {self.RealTime}')
        
        if self.RealTime:
            self.IP = config['Info']['IP']
            self.port = config['Info']['port']
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.bind((self.IP,self.port))
            self.client.listen(5)
            print(f'Server start at {self.IP}:{self.port}')
            print('waiting for connection...')
            conn, addr = self.client.accept()
            print(f'Connected by {addr}')
            self.conn = conn
            self.addr = addr

            info = 'XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nsimulator\n\0'
            self.conn.send(info.encode('utf-8'))
            
            data = self.conn.recv(1024)
            print(data.decode('utf-8'))
            print('shaking hand done')
        
    def reset(self, init_info):
        '''
        init_info : Dict
        all data type : str
        {'time':'10.24'
         'Name1':[longitude, latitude, altitude, roll, pitch, yaw],
         'Name2':[longitude, latitude, altitude, roll, pitch, yaw],}
        '''
        #assume reset at
        self.data = 'FileType=' + self.FileType + '\n' + \
                    'FileVersion=' + self.FileVersion + '\n' + \
                    '0,ReferenceTime=' + self.ReferenceTime + '\n'
        if self.RealTime:
            self.conn.send(self.data.encode('utf-8'))


        self.data += '#' + init_info['time'] + '\n'
        for name,pos in init_info.items():
            if name=='time':
                continue
            #red1,T=5.3320031|4.9645909|1997.98|0.0|0.3|0.2,
            info = name+',T='+'|'.join(pos)+','
            #red1,T=5.3320031|4.9645909|1997.98|0.0|0.3|0.2,Type=Air+FixedWing,Color=Blue,Name=F-22A
            info += 'Type='+self.plane_dict[name]['Type']+','
            info += 'Color='+self.plane_dict[name]['Color']+','
            info += 'Name='+self.plane_dict[name]['Model']
            info += '\n'
            self.data += info
        self.T = float(init_info['time'])
        
        if self.RealTime:
            self.conn.send(self.data.encode('utf-8'))
            
        
        
    def record(self, state_all, message = None):
        '''
        state_all : Dict
        all data type : str
        {'Name1':[longitude, latitude, altitude, roll, pitch, yaw],
         'Name2':[longitude, latitude, altitude, roll, pitch, yaw],}
        '''
        #assume reset at 
        self.T += self.dt
        info = '#' + str(self.T) + '\n'
        for name,pos in state_all.items():
            #red1,T=5.3320031|4.9645909|1997.98|0.0|0.3|0.2,
            info += name+',T='+'|'.join(pos)+'\n'
            
        self.data += info
        if self.RealTime:
            self.conn.send(info.encode('utf-8'))
        #'2022-07-06T08:39:21.153Z'

        # 解析时间字符串为 datetime 对象
        time_obj = datetime.strptime(self.ReferenceTime, '%Y-%m-%dT%H:%M:%S.%fZ')

        # 转换为时间戳（秒）
        timestamp_seconds = time_obj.timestamp()

        timestamp_seconds+=1/60
        time_obj = datetime.fromtimestamp(timestamp_seconds)

        self.ReferenceTime = time_obj.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
    def insert(self, info):
        self.data += info
            
        
    def export(self, path):
        if self.RealTime:
            time.sleep(10)
            self.client.close()
        with open(file=path, mode='w+', encoding='utf-8') as f:
            f.write(self.data)


if __name__=='__main__':
    test = TacviewRecorder(default_config)
    test.reset({'time':'0','101':['36','37','2000','0','0','0'],'102':['36','37.001','2000','0','90','0']})
    test.record({'101':['36','37.001','2000','0','0','0'],'102':['36','37.002','2000','0','90','0']})
    test.record({'101':['36','37.002','2000','0','0','0'],'102':['36','37.003','2000','0','90','0']})
    test.export('./data.txt.acmi')
        