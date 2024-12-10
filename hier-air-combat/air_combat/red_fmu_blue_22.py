from env.env_client import EnvClient
from tacview import TacviewRecorder
import numpy as np
import math

from math import degrees as r2d
from math import radians as d2r
import matplotlib.pyplot as plt

recorder_settings = { 
    'Env':
        {'FileType' : 'text/acmi/tacview',
         'FileVersion' : '2.1',
         'ReferenceTime' : '2022-07-06T08:39:21.153Z',
         'DeltaTime' : '0.2'},
    'Plane':
        {
         '101' : {'Type':'Air+FixedWing', 'Model':'F-22A', 'Color':'Blue'},
         '201' : {'Type':'Air+FixedWing', 'Model':'F-22A', 'Color':'Red'},
         '202' : {'Type':'Air+FixedWing', 'Model':'F-22A', 'Color':'Red'},
         '203' : {'Type':'Air+FixedWing', 'Model':'F-22A', 'Color':'Red'},
         '204' : {'Type':'Air+FixedWing', 'Model':'F-22A', 'Color':'Red'},
         '205' : {'Type':'Air+FixedWing', 'Model':'F-22A', 'Color':'Red'},
         '206' : {'Type':'Air+FixedWing', 'Model':'F-22A', 'Color':'Red'},},
    'RealTime':False,
    'Info':{'IP':'localhost','port':7788}
    }
# fmu model init form 
# {'initial_altitude_ft': 20000, 'initial_longitude': 119.999, 'initial_latitude': 25.0, 'initial_heading_degree': 0, 'initial_mach': 0.547}
# jsb model init form 
# {'initial_altitude_ft': 20000, 'initial_longitude_geoc_deg': 119.999, 'initial_latitude_geod_deg': 25.0, 'initial_heading_deg': 0, 'initial_mach': 0.547}


init_dict = {'101': {'initial_altitude_ft': 20000, 'initial_longitude_geoc_deg': 120,
                         'initial_latitude_geod_deg': 26, 'initial_heading_deg': 180, 'initial_mach': 0.8},
             '201': {'initial_altitude_ft': 20000, 'initial_longitude': 119.999,
                         'initial_latitude': 25.0, 'initial_heading_degree': 0, 'initial_mach': 0.547},
             '202': {'initial_altitude_ft': 20000, 'initial_longitude': 119.9995,
                         'initial_latitude': 25.0, 'initial_heading_degree': 0, 'initial_mach': 0.547},
             '203': {'initial_altitude_ft': 20000, 'initial_longitude': 120,
                         'initial_latitude': 25.0, 'initial_heading_degree': 0, 'initial_mach': 0.547},
             '204': {'initial_altitude_ft': 20000, 'initial_longitude': 120.0005,
                         'initial_latitude': 25.0, 'initial_heading_degree': 0, 'initial_mach': 0.547},
             '205': {'initial_altitude_ft': 20000, 'initial_longitude': 120.001,
                         'initial_latitude': 25.0, 'initial_heading_degree': 0, 'initial_mach': 0.547},
             '206': {'initial_altitude_ft': 20000, 'initial_longitude': 120.0015,
                         'initial_latitude': 25.0, 'initial_heading_degree': 0, 'initial_mach': 0.547},
             }
#only can set longitude, latitude and heading degree, altitude constant to 20000, init mach constant to 0.547

queue = {'201':[-60,-40,-40],
         '202':[-30,-20,-20],
         '203':[0,0,0],
         '204':[30,-20,-20],
         '205':[60,-40,-40],
         '206':[90,-60,-60],}

class PID():
    def __init__(self,mode, pid, config):        
        self.mode = mode
        self.P = pid['P']
        self.I = pid['I']
        self.D = pid['D']
        
        self.delta = [0, 0]
        self.delta_ = [0, 0]
        self.accu = 0
        
        if self.mode == 'integral separated PID':
            self.I_threshold = config['threshold']
        self.config = config

        self.gamma = config['gamma']
        
        
    def calculate(self, target, current):
        
        self.delta[1] = self.delta[0]
        self.delta_[1] = self.delta_[0]
        self.delta[0] = target - current
        if self.config.get('angle',0)==1:
            if self.delta[0]<-180:
                self.delta[0] += 360
            elif self.delta[0]>180:
                self.delta[0] -= 360
        self.delta_[0] = self.delta[1] - self.delta[0]
        if self.mode == 'integral separated PID' and abs(self.delta[0])>self.I_threshold:
            self.accu = 0 
        else:
            self.accu = self.delta[0] + self.gamma*self.accu
            
        if self.mode == 'PD':
            return self.P*self.delta[0] + self.D*self.delta_[0]
        elif self.mode == 'PID':
            return self.P*self.delta[0] + self.I*self.accu + self.D*self.delta_[0]
        elif self.mode == 'integral separated PID':
            return self.P*self.delta[0] + self.I*self.accu + self.D*self.delta_[0]
        

class Approach():
    def __init__(self):
        self.env = EnvClient(init_dict)
        
        self.plane_list = list(init_dict)
        self.plane_destroyed = {}
        for plane in self.plane_list:
            self.plane_destroyed[plane] = 0
        
        self.observation = {}
        self.trajectory = {}
        
        self.T = 0.2
        #aileron_compensate
        self.K = 0.1
        
        self.state_last = {}
        self.h_PID = {}
        self.pitch_PID = {}
        self.speed_PID = {}
        self.psi_PID = {}
        self.roll_PID = {}
        for plane in self.plane_list:
            self.state_last[plane] = {'elevator_last' : 0, 'throttle_last' : 0.4, 'aileron_last' : 0}
            self.h_PID[plane] = PID('integral separated PID', {'P':1/22.5, 'I':0.01, 'D':-0.05}, {'threshold':100,'gamma':0.3})
            self.pitch_PID[plane] = PID('PD', {'P':-0.05, 'I':0, 'D':0.01}, {'gamma':0.5,'angle':1})
            self.speed_PID[plane] = PID('PD', {'P':1, 'I':0, 'D':-0.4}, {'threshold':100,'gamma':0.5})
            self.psi_PID[plane] = PID('PID', {'P':2, 'I':0.01, 'D':-0.05}, {'threshold':1,'gamma':0.5,'angle':1})
            self.roll_PID[plane] = PID('integral separated PID', {'P':-0.01, 'I':0.01, 'D':0.02}, {'threshold':1,'gamma':0.5,'angle':1})

        self.reset(init_dict)
    
    def reset(self,init_info):
        raw_obs = self.env.reset(init_info)
        self._parse_observation(raw_obs)
        
        return self.observation
    
    def _parse_observation(self,raw_obs):
        for plane,info in raw_obs.items():
            if self.plane_destroyed[plane]==1:
                continue
            self.observation[plane] = {'longitude' : info['position_long_gc_deg'],
                                  'latitude' : info['position_lat_geod_deg'],
                                  'altitude' : info['position_h_sl_ft']/3.28,
                                  'roll' : r2d(info['attitude_roll_rad']),
                                  'pitch' : r2d(info['attitude_pitch_rad']),
                                  'velocity' : info['velocities_mach']}
            
            if info['attitude_psi_deg']<=180:
                self.observation[plane]['psi'] = -info['attitude_psi_deg']
            elif info['attitude_psi_deg']<=360:
                self.observation[plane]['psi'] = 360-info['attitude_psi_deg']


            self.trajectory[plane] = [str(self.observation[plane]['longitude']), str(self.observation[plane]['latitude']), str(self.observation[plane]['altitude']+5000),
                                      str(self.observation[plane]['roll']), str(self.observation[plane]['pitch']), str(-self.observation[plane]['psi'])]
        

    def step(self,actions):
        raw_obs = self.env.step(actions)
        self._parse_observation(raw_obs)
        #self.recorder.record(self.trajectory)
        return self.observation
    
    def get_obs(self):
        return self.observation
    
    def select_action(self, target_info):
        actions = {}
        for plane in self.plane_list:
            aileron, aileron_compensate = self.psi_control(plane, target_info[plane]['target_psi'])
            elevator, speed_compensate = self.height_control(plane, target_info[plane]['target_height'], aileron_compensate)
            throttle = self.speed_control(plane, target_info[plane]['target_speed'], speed_compensate)
            actions[plane] = {'aileron':aileron, 'elevator':elevator, 'throttle':throttle}
        return actions
    
    def height_control(self, plane, target_height, aileron_compensate):
        elevator = 0
        
        #outer PID target_h to target_pitch
        #integral separated PID control
        
        current_height = self.observation[plane]['altitude']
        speed_compensate = -(target_height-current_height)*0.00013
        target_pitch = self.h_PID[plane].calculate(target_height, current_height)
        target_pitch += aileron_compensate
        target_pitch = np.clip(target_pitch,-30, 35)
        
        
        
        #inner PID target_pitch to elvator
        current_pitch = self.observation[plane]['pitch']
        elevator_delta = self.pitch_PID[plane].calculate(target_pitch, current_pitch)

        elevator_delta = np.clip(elevator_delta,-0.2, 0.2)

        elevator = np.clip(self.state_last[plane]['elevator_last']+elevator_delta, -1, 1)
        
        self.state_last[plane]['elevator_last'] = elevator
        
        if plane == '101':
            print(f'4============{target_pitch}')
            print(f'5============{current_pitch}')
            print(f'6============{target_height}')
            print(f'7============{current_height}')
        
        return elevator, speed_compensate 
    
    def speed_control(self, plane, target_speed, speed_compensate):

        current_speed = self.observation[plane]['velocity']
        delta_throttle = self.speed_PID[plane].calculate(target_speed-speed_compensate, current_speed)
        delta_throttle = np.clip(delta_throttle, -0.05, 0.05)
        throttle = np.clip(self.state_last[plane]['throttle_last']+delta_throttle,0.2,1)
        
        self.state_last[plane]['throttle_last'] = throttle
        
        return throttle
            

    def psi_control(self, plane, target_psi):
        
        aileron = 0
        
        #outer PID target_yaw to target_roll
        #integral separated PID control
        current_psi = -self.observation[plane]['psi']
        
        target_roll = self.psi_PID[plane].calculate(target_psi, current_psi)
        target_roll = np.clip(target_roll, -40, 40)
        if plane == '101':
            print(f'1============{target_roll}')
            print(f'2============{target_psi}')
            print(f'3============{current_psi}')
        
        aileron_compensate = self.K*abs(target_roll)
        
        #inner PID target_pitch to aileron
        current_roll = self.observation[plane]['roll']
        aileron_delta = self.roll_PID[plane].calculate(target_roll, current_roll) 
        
        aileron_delta = np.clip(aileron_delta,-0.05, 0.05)

        aileron = np.clip(self.state_last[plane]['aileron_last']+aileron_delta, -1, 1) 
        self.state_last[plane]['aileron_last'] = aileron
        
# =============================================================================
#         print(f'plane : {plane}')
#         print(f'current psi={current_psi}, target_psi={target_psi}')
#         print(f'current roll={current_roll}, target_roll={target_roll}')
# =============================================================================
        
        return aileron, aileron_compensate    
    
class formation():
    def __init__(self, queue, leader):
        self.queue = queue
        self.plane_list = list(queue)
        self.leader = leader
        self.current_state = {}
        self.relative_state = {}
        self.dx_PID = {}
        for plane in self.plane_list:
            self.current_state[plane] = np.zeros(3)
            self.relative_state[plane] = np.zeros(3)
            self.dx_PID[plane] = PID('integral separated PID', {'P':1/5, 'I':0.05, 'D':-0.05}, {'threshold':20,'gamma':0.7})

    
    def transform(self,state):
        for plane in self.plane_list:
            self.current_state[plane][0] = state[plane]['longitude']
            self.current_state[plane][1] = state[plane]['latitude'] 
            self.current_state[plane][2] = state[plane]['altitude']
        center = self.current_state[self.leader]
        for plane in self.plane_list:
            self.relative_state[plane] = self.current_state[plane] - center
            self.relative_state[plane][0] *= 111194.926644*math.cos(d2r(center[1]))
            self.relative_state[plane][1] *= 111194.926644
        
    def decision(self, state, center_height, center_speed):
        target_info = {}
        self.transform(state)
        for plane in self.plane_list:
            target_info[plane] = {}

            target_info[plane]['target_height'] = center_height + self.queue[plane][2]            

            dx = self.relative_state[plane][0]-self.queue[plane][0]
            dy = self.relative_state[plane][1]-self.queue[plane][1]
            target_info[plane]['target_speed'] = np.clip(center_speed-dy/2000, 0.5, 0.9) 

            target_rel_x = self.queue[plane][0]
            current_rel_x = self.relative_state[plane][0]
            target_info[plane]['target_psi'] = np.clip(self.dx_PID[plane].calculate(target_rel_x, current_rel_x), -30, 30)

        return target_info
    
class Missile():
    def __init__(self, target, init_state, name):
        self.target = target
        self.longitude = init_state['longitude']
        self.latitude = init_state['latitude']
        self.altitude = init_state['altitude']
        self.name = name
        self.T = 0.2
        self.dis_T = 2.5*340*0.2
        self.init_flag = 1
        
        self.hit = 0
        
    def update(self, obs):
        target_lon = obs[self.target]['longitude']
        target_lat = obs[self.target]['latitude']
        target_alt = obs[self.target]['altitude']+5000
        delta_longitude = abs(target_lon-self.longitude)
        delta_latitude = abs(target_lat-self.latitude)
        delta_altitude = abs(target_alt-self.altitude)
        middle_latitude = (target_lat-self.latitude)/2
        dis = ((delta_longitude*111194*math.cos(middle_latitude))**2+(delta_latitude*111194)**2+(delta_altitude)**2)**0.5
        if dis<self.dis_T:
            self.hit = 1
        percent = self.dis_T/dis
        new_lon = target_lon*percent+self.longitude*(1-percent)
        new_lat = target_lat*percent+self.latitude*(1-percent)
        new_alt = target_alt*percent+self.altitude*(1-percent)

        self.longitude = new_lon
        self.latitude = new_lat
        self.altitude = new_alt

        return new_lon, new_lat, new_alt
    
    def get_tacview(self):
        if self.init_flag==1:
            self.init_flag = 0
            return self.name+',T='+str(self.longitude)+'|'+str(self.latitude)+'|'+str(self.altitude)+',Type=Weapon+Missile,Color=Blue,Name=AIM_120\n'
        else:
            return self.name+',T='+str(self.longitude)+'|'+str(self.latitude)+'|'+str(self.altitude)+'\n'
        
                                                                                     
        
    
def GetDistance(obs, blue_state):
    Min = 1000000
    for plane in obs:
        if plane=='101':
            continue
        red_state = obs[plane]
        delta_longitude = abs(red_state['longitude']-blue_state['longitude'])
        delta_latitude = abs(red_state['latitude']-blue_state['latitude'])
        middle_latitude = (red_state['latitude']-blue_state['latitude'])/2
        dis = ((delta_longitude*111194*math.cos(middle_latitude))**2+(delta_latitude*111194)**2)**0.5
        if dis<Min:
            Min = dis
    return Min
                
    
if __name__=="__main__":
    form = formation(queue,'203')
    test = Approach()  
    recorder = TacviewRecorder(recorder_settings)
    init_info = {'time':'0'}
    init_info.update(test.trajectory)
    recorder.reset(init_info)
    obs = test.get_obs()
    blue_state = obs['101']
    print('start !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    target_info = {}
    missile_list = []
    missile_flag = [1]*6
    
    
    dis = GetDistance(obs, blue_state)
    
    while dis>100000:

        if dis>100000:
            #红方编队， 蓝方直飞
            red_target = form.decision(obs, 6000, 0.7)
            blue_target = {'101': {'target_height' : 6000, 'target_speed' : 0.9, 'target_psi' : 180}}
            target_info.update(red_target)
            target_info.update(blue_target)
        elif dis>85000:
            #红方分散， 蓝方发弹
            target_info = {'101':{'target_height' : 6500, 'target_speed' : 0.9, 'target_psi' : 180},
                           '201':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : -40},
                           '202':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : -20},
                           '203':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 0},
                           '204':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 20},
                           '205':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 40},
                           '206':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 50}}
            #发射两个导弹，间隔2s
            if missile_flag[0]==1:
                missile_list.append(Missile('203', blue_state, '301'))
                missile_flag[0] = 0
                missile_interval = 10
            elif missile_flag[1]==1:
                if missile_interval <=0:
                    missile_list.append(Missile('203', blue_state, '302'))
                    missile_flag[1] = 0
                else:
                    missile_interval-=1
        elif dis>80000:
            target_info = {'101':{'target_height' : 6500, 'target_speed' : 0.9, 'target_psi' : 180},
                           '201':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 0},
                           '202':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 0},
                           '203':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 0},
                           '204':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 0},
                           '205':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 0},
                           '206':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 0}}
        elif dis>54000:
            target_info = {'101':{'target_height' : 6500, 'target_speed' : 0.9, 'target_psi' : 180},
                           '201':{'target_height' : 5500, 'target_speed' : 0.75, 'target_psi' : 0},
                           '202':{'target_height' : 6500, 'target_speed' : 0.75, 'target_psi' : 0},
                           '203':{'target_height' : 6000, 'target_speed' : 0.75, 'target_psi' : 0},
                           '204':{'target_height' : 6500, 'target_speed' : 0.75, 'target_psi' : 0},
                           '205':{'target_height' : 6000, 'target_speed' : 0.75, 'target_psi' : 0},
                           '206':{'target_height' : 5500, 'target_speed' : 0.75, 'target_psi' : 0}}
        else:
            target_info = {'101':{'target_height' : 6500, 'target_speed' : 0.9, 'target_psi' : 180},
                           '201':{'target_height' : 5500, 'target_speed' : 0.9, 'target_psi' : 0},
                           '202':{'target_height' : 6500, 'target_speed' : 0.9, 'target_psi' : 0},
                           '203':{'target_height' : 6000, 'target_speed' : 0.9, 'target_psi' : 0},
                           '204':{'target_height' : 6500, 'target_speed' : 0.9, 'target_psi' : 0},
                           '205':{'target_height' : 6000, 'target_speed' : 0.9, 'target_psi' : 0},
                           '206':{'target_height' : 5500, 'target_speed' : 0.9, 'target_psi' : 0}}
            #发射两个导弹，间隔2s
            if missile_flag[2]==1:
                missile_list.append(Missile('201', blue_state, '303'))
                missile_list.append(Missile('206', blue_state, '304'))
                missile_flag[2] = 0
                missile_flag[3] = 0
                missile_interval = 20
            elif missile_flag[4]==1:
                if missile_interval <=0:
                    missile_list.append(Missile('201', blue_state, '305'))
                    missile_list.append(Missile('206', blue_state, '306'))
                    missile_flag[4] = 0
                    missile_flag[5] = 0
                else:
                    missile_interval-=1
            


        actions = test.select_action(target_info) 
        print(actions['101'])
        obs = test.step(actions)
        blue_state = obs['101']
        recorder.record(test.trajectory)
        
# =============================================================================
#         blue_state['latitude'] -= 1.5*340*0.2/111194
#         recorder.insert('101,T=120|'+str(blue_state['latitude'])+'|11000|0|180|0\n')
# =============================================================================
        
        
        dis = GetDistance(obs, blue_state)

        print(dis)
        
        for missile in missile_list:
            missile.update(obs)
            missile_info = missile.get_tacview()
            recorder.insert(missile_info)
        #击毁后，导弹移除当前导弹队列，飞机移除approach飞机队列
        #多枚导弹瞄准同一敌人，不要重复pop
        for ind,missile in enumerate(missile_list):
            if missile.hit :
                missile_list.pop(ind)
                recorder.insert('-'+missile.name+'\n')
                recorder.insert('-'+missile.target+'\n')
                test.plane_destroyed[missile.target]=1
                if test.plane_destroyed[missile.target]==0:
                    test.trajectory.pop(missile.target)
                
            

# =============================================================================
#     for i in range(200):
#         print(f'i={i}')
#         target_info = {'101':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 180},
#                        '201':{'target_height' : 5960, 'target_speed' : 0.7, 'target_psi' : 10},
#                        '202':{'target_height' : 5980, 'target_speed' : 0.7, 'target_psi' : 0},
#                        '203':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 0},
#                        '204':{'target_height' : 5980, 'target_speed' : 0.7, 'target_psi' : 0},
#                        '205':{'target_height' : 5960, 'target_speed' : 0.7, 'target_psi' : 0},
#                        '206':{'target_height' : 5960, 'target_speed' : 0.7, 'target_psi' : 0}}
#         target_info.update(form.decision(obs, 6000, 0.7))
# 
#         actions = test.select_action(target_info)
#         obs = test.step(actions)
#     long, lat, alt = obs['101']['longitude'], obs['101']['latitude'], obs['101']['altitude']
#     test.recorder.data += '301,T='+str(long)+'|'+str(lat)+'|'+str(alt)+'Type=Weapon+Missile,Color=Blue,Name=AIM_120\n'
#     for i in range(200,300):
#         print(f'i={i}')
#         target_info = {'101':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 180},
#                        '201':{'target_height' : 5960, 'target_speed' : 0.7, 'target_psi' : -40},
#                        '202':{'target_height' : 5980, 'target_speed' : 0.7, 'target_psi' : -20},
#                        '203':{'target_height' : 6000, 'target_speed' : 0.7, 'target_psi' : 0},
#                        '204':{'target_height' : 5980, 'target_speed' : 0.7, 'target_psi' : 20},
#                        '205':{'target_height' : 5960, 'target_speed' : 0.7, 'target_psi' : 40},
#                        '206':{'target_height' : 5960, 'target_speed' : 0.7, 'target_psi' : 60}}
# 
#         actions = test.select_action(target_info)
#         
#         long1, lat1, alt1 = obs['101']['longitude'], obs['101']['latitude'], obs['101']['altitude']
#         long2, lat2, alt2 = obs['203']['longitude'], obs['203']['latitude'], obs['203']['altitude']
#         long, lat, alt = (3-i*0.01)*long1+(i*0.01-2)*long2, (3-i*0.01)*lat1+(i*0.01-2)*lat2, (3-i*0.01)*alt1+(i*0.01-2)*alt2
#         test.recorder.data += '301,T='+str(long)+'|'+str(lat)+'|'+str(alt)+'\n'
#         
#         
#         
#         
#         
#         #actions['201'] = {'aileron':0.1, 'elevator':0.1, 'throttle':0.5}
#         
#         obs = test.step(actions)
# =============================================================================

    recorder.export('./temp.txt.acmi')
    
    