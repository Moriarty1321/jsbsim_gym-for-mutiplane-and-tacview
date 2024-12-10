import pdb

import numpy as np
import math

from math import degrees as r2d
from math import radians as d2r

#from blue_rule import Rule_Agent

#only can set longitude, latitude and heading degree, altitude constant to 20000, init mach constant to 0.547


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
        if config.get('angle'):
            self.angle = True
        else:
            self.angle = False

        self.gamma = config['gamma']


    def calculate(self, target, current):

        self.delta[1] = self.delta[0]
        self.delta_[1] = self.delta_[0]
        self.delta[0] = target - current
        if self.angle:
            if self.delta[0]>180:
                self.delta[0] -= 360
            elif self.delta[0]<-180:
                self.delta[0] += 360
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


class Rule_Agent():
    def __init__(self,plane_dict,relationship):
        ##以往初始化变量一览
        self.plane_list = list(plane_dict.keys())
        self.relationship = relationship
        self.observation = {}
        self.state = {}
        self.trajectory = {}

        self.T = 0.05
        #aileron_compensate
        self.K = 0.1

        self.state_last = {}
        self.action_last = {}
        self.h_PID = {}
        self.pitch_PID = {}
        self.speed_PID = {}
        self.psi_PID = {}
        self.roll_PID = {}
        for plane in self.plane_list:
            self.action_last[plane] = {'aileron': 0 , 'elevator':0 , 'throttle': 0.4}
            self.state[plane] = {
                'longitude_last':0,
                'latitude_last': 0,
                'altitude_last': 0,
            }
            if plane_dict[plane] == 'f22':
                self.h_PID[plane] = PID('integral separated PID', {'P':1/40, 'I':1/50, 'D':-0.03}, {'threshold':100,'gamma':0.6})
                self.pitch_PID[plane] = PID('PD', {'angle':True,'P':-1/35, 'I':0, 'D':0.02}, {'gamma':0.5})
                self.speed_PID[plane] = PID('PD', {'P':0.5, 'I':0, 'D':-0.4}, {'threshold':100,'gamma':0.5})
                self.psi_PID[plane] = PID('PID', {'P':2, 'I':0.01, 'D':-0.05}, {'angle':True,'threshold':1,'gamma':0.5})
                self.roll_PID[plane] = PID('integral separated PID', {'angle':True,'P':1/50, 'I':0.01, 'D':0.02}, {'threshold':5,'gamma':0.5})
            elif plane_dict[plane] == 'fmu_new':
                self.h_PID[plane] = PID('integral separated PID', {'P': 1 / 40, 'I': 1 / 80, 'D': -0.1},
                                        {'threshold': 100, 'gamma': 0.4})
                self.pitch_PID[plane] = PID('PD', {'angle': True, 'P': -1 / 70, 'I': 0, 'D': 0.02}, {'gamma': 0.5})
                self.speed_PID[plane] = PID('PD', {'P': 0.5, 'I': 0, 'D': -0.4}, {'threshold': 100, 'gamma': 0.5})
                self.psi_PID[plane] = PID('PID', {'P': 4, 'I': 0.01, 'D': -0.05},
                                          {'angle': True, 'threshold': 1, 'gamma': 0.5})
                self.roll_PID[plane] = PID('PD', {'angle': True, 'P': -1 / 60, 'I': 0.01, 'D': 0.03},
                                           {'threshold': 5, 'gamma': 0.5})
            elif plane_dict[plane] == 'fmu_old':
                self.h_PID[plane] = PID('integral separated PID', {'P':1/22.5, 'I':0.01, 'D':-0.05}, {'threshold':100,'gamma':0.3})
                self.pitch_PID[plane] = PID('PD', {'angle':True,'P':-0.05, 'I':0, 'D':0.01}, {'gamma':0.5})
                self.speed_PID[plane] = PID('PD', {'P':1, 'I':0, 'D':-0.4}, {'threshold':100,'gamma':0.5})
                self.psi_PID[plane] = PID('PID', {'P':2, 'I':0.01, 'D':-0.05}, {'angle':True,'threshold':1,'gamma':0.5})
                self.roll_PID[plane] = PID('integral separated PID', {'angle':True,'P':-0.01, 'I':0.01, 'D':0.02}, {'threshold':1,'gamma':0.5})
            elif plane_dict[plane] == 'bisai0110':
                self.h_PID[plane] = PID('integral separated PID', {'P': 1 / 40, 'I': 1 / 10, 'D': -0.1},
                                        {'threshold': 100, 'gamma': 0.4})
                self.pitch_PID[plane] = PID('PD', {'angle': True, 'P': -0.1, 'I': 0.05, 'D': 0.02}, {'gamma': 0.5})
                self.speed_PID[plane] = PID('PD', {'P': 0.5, 'I': 0, 'D': -0.4}, {'threshold': 100, 'gamma': 0.5})
                self.psi_PID[plane] = PID('PID', {'P': 4, 'I': 0.01, 'D': -0.05},
                                          {'angle': True, 'threshold': 1, 'gamma': 0.5})
                self.roll_PID[plane] = PID('PD', {'angle': True, 'P': 1 / 50, 'I': 0.01, 'D': 0.02},
                                           {'threshold': 5, 'gamma': 0.5})

            elif plane_dict[plane] == 'New0111_red':
                self.h_PID[plane] = PID('integral separated PID', {'P': 1 /40, 'I': 1 / 30, 'D': -0.2},   ###'P': 1 /40, 'I': 1 / 30, 'D': -0.2
                                        {'threshold': 100, 'gamma': 0.4})
                self.pitch_PID[plane] = PID('PD', {'angle': True, 'P': -0.2, 'I': 1 / 40, 'D': 0.02}, {'gamma': 0.5})
                self.speed_PID[plane] = PID('PD', {'P': 0.5, 'I': 0, 'D': -0.4}, {'threshold': 100, 'gamma': 0.5})
                self.psi_PID[plane] = PID('PID', {'P': 2, 'I': 0.01, 'D': -0.05},
                                          {'angle': True, 'threshold': 1, 'gamma': 0.5})
                self.roll_PID[plane] = PID('PD', {'angle': True, 'P': 1 / 50, 'I': 0.01,'D': 0.01},    ###'P': -0.001, 'I': 0,'D': 0.02
                                           {'threshold': 5, 'gamma': 0.5})

            elif plane_dict[plane] == 'New0111_blue':
                self.h_PID[plane] = PID('integral separated PID', {'P': 1 /20, 'I': 1 / 30, 'D': -0.3},
                                        {'threshold': 100, 'gamma': 0.4})
                self.pitch_PID[plane] = PID('PD', {'angle': True, 'P': -0.025, 'I': 0.05, 'D': 0.02}, {'gamma': 0.5})
                self.speed_PID[plane] = PID('PD', {'P': 0.5, 'I': 0, 'D': -0.4}, {'threshold': 100, 'gamma': 0.5})
                self.psi_PID[plane] = PID('PID', {'P': 2, 'I': 0.01, 'D': -0.05},
                                          {'angle': True, 'threshold': 1, 'gamma': 0.5})
                self.roll_PID[plane] = PID('PD', {'angle': True, 'P': 1 / 50, 'I': 0.01, 'D': 0.02},
                                           {'threshold': 5, 'gamma': 0.5})
            elif plane_dict[plane] == 'U99':
                self.h_PID[plane] = PID('integral separated PID', {'P': 1 /40, 'I': 1 / 30, 'D': -0.3},
                                        {'threshold': 100, 'gamma': 0.4})
                self.pitch_PID[plane] = PID('PD', {'angle': True, 'P': -0.025, 'I': 0.05, 'D': 0.02}, {'gamma': 0.5})
                self.speed_PID[plane] = PID('PD', {'P': 0.5, 'I': 0, 'D': -0.4}, {'threshold': 100, 'gamma': 0.5})
                self.psi_PID[plane] = PID('PID', {'P': 2, 'I': 0.01, 'D': -0.05},
                                          {'angle': True, 'threshold': 1, 'gamma': 0.5})
                self.roll_PID[plane] = PID('PD', {'angle': True, 'P': -0.005, 'I': 0.01, 'D': 0.01},
                                           {'threshold': 5, 'gamma': 0.5})
            else:
                print("Unknown type")
                raise ValueError
    def _parse_observation(self, raw_obs):   #0108环境下进行环境状态变量的提取
        # combine_obs_dict = {**raw_red_obs.my_planes, **raw_blue_obs.my_planes}
        for plane,info in raw_obs.items():
            if plane not in self.plane_list:
                continue
            self.state[plane].update({'longitude': info["lon"],
                                     'latitude': info["lat"],
                                     'altitude': info["height"],
                                     'roll': info["roll"],
                                     'pitch': info["pitch"],
                                     'velocity': info["mach"],
                                     })
            # pdb.set_trace()
            if info["yaw"]<=180:
                self.state[plane]['psi'] = info["yaw"]
            elif info["yaw"]<=360:
                self.state[plane]['psi'] = info["yaw"]-360

            delta_north_m = (self.state[plane]['latitude']-self.state[plane]['latitude_last'])*111194
            delta_east_m = (self.state[plane]['longitude']-self.state[plane]['longitude_last']) * 111194 * math.cos(d2r(self.state[plane]['latitude']))
            delta_up_m = (self.state[plane]['altitude']-self.state[plane]['altitude_last'])
            self.state[plane]['v_north'] = delta_north_m/self.T
            self.state[plane]['v_east'] = delta_east_m / self.T
            self.state[plane]['v_up'] = delta_up_m / self.T

            self.state[plane]['longitude_last'] = info["lon"]
            self.state[plane]['latitude_last'] = info["lat"]
            self.state[plane]['altitude_last'] = info["height"]   #修改这个变量数

            self.trajectory[plane] = [str(self.state[plane]['longitude']),
                                      str(self.state[plane]['latitude']),
                                      str(self.state[plane]['altitude'] ),
                                      str(self.state[plane]['roll']), str(self.state[plane]['pitch']),
                                      str(info["yaw"])]

        def _get_relative_info(plane_ego, plane_enm):

            def _get_distance(lon1, lat1, alt1, lon2, lat2, alt2):
                # 用于计算距离
                delta_longitude = abs(lon1 - lon2)
                delta_latitude = abs(lat1 - lat2)
                middle_latitude = (lat1 + lat2) / 2
                dis = (
                              (delta_longitude * 111194 * math.cos(d2r(middle_latitude))) ** 2
                              + (delta_latitude * 111194) ** 2
                              + (alt1 - alt2) ** 2
                      ) ** 0.5

                return dis / 1000

            distance = _get_distance(self.state[plane_ego]['longitude'], self.state[plane_ego]['latitude'],
                                     self.state[plane_ego]['altitude'],
                                     self.state[plane_enm]['longitude'], self.state[plane_enm]['latitude'],
                                     self.state[plane_enm]['altitude'])
            distance_xy = _get_distance(self.state[plane_ego]['longitude'], self.state[plane_ego]['latitude'], 0,
                                        self.state[plane_enm]['longitude'], self.state[plane_enm]['latitude'], 0)
            # pdb.set_trace()
            # calculate angle between vector a and b radian(0-pi)
            def _calculate_angle(a, b):
                """
                """
                # pdb.set_trace()
                a = np.array(a)
                b = np.array(b)
                a_norm = (a @ a) ** 0.5
                b_norm = (b @ b) ** 0.5
                dot_ab = a @ b
                # pdb.set_trace()
                cos_value = dot_ab / (a_norm * b_norm + 1e-6)
                cos_value = np.clip(round(cos_value, 10), -1, 1)

                return math.acos(cos_value)

            delta_longitude = self.state[plane_enm]['longitude'] - self.state[plane_ego]['longitude']
            delta_latitude = self.state[plane_enm]['latitude'] - self.state[plane_ego]['latitude']
            delta_altitude = self.state[plane_enm]['altitude'] - self.state[plane_ego]['altitude']
            delta_velocity = self.state[plane_enm]['velocity'] - self.state[plane_ego]['velocity']

            baseline = [delta_longitude * 111194 * math.cos(
                d2r(self.state[plane_enm]['latitude'] / 2 + self.state[plane_ego]['latitude'] / 2)),
                        delta_latitude * 111194, delta_altitude]
            # dong bei tian ------ longitude latitude altitude
            ego_vector = [self.state[plane_ego]['v_east'], self.state[plane_ego]['v_north'],
                          self.state[plane_ego]['v_up']]
            enm_vector = [self.state[plane_enm]['v_east'], self.state[plane_enm]['v_north'],
                          self.state[plane_enm]['v_up']]
            lock = _calculate_angle(baseline, ego_vector)  # 0-pi
            escape = _calculate_angle(baseline, enm_vector)  # 0-pi

            lock_xy = _calculate_angle(baseline[:2], ego_vector[:2])
            if np.cross(baseline[:2], ego_vector[:2]) <0:
                lock = -lock
                lock_xy = -lock_xy

            # pdb.set_trace()
            # left -1 right 1
            # sig = np.sign(ego_vector[1] * baseline[0] - ego_vector[0] * baseline[1])
            # if sig != 0:
            #     lock_xy *= sig
            escape_xy = _calculate_angle(baseline[:2], enm_vector[:2])
            sig = np.sign(enm_vector[1] * baseline[0] - enm_vector[0] * baseline[1])
            if sig != 0:
                escape_xy *= sig
            # angle of baseline and horiz
            # pdb.set_trace()
            anglebase = math.atan(delta_altitude / (distance_xy+1e-6) / 1000)
            # angle of ego speed and horiz
            angleego = math.atan(self.state[plane_ego]['v_up'] / (
                        (self.state[plane_ego]['v_east'] ** 2 + self.state[plane_ego]['v_north'] ** 2) ** 0.5 + 1e-6))
            angleenm = math.atan(self.state[plane_enm]['v_up'] / (
                        (self.state[plane_enm]['v_east'] ** 2 + self.state[plane_enm]['v_north'] ** 2) ** 0.5 + 1e-6))
            lock_z = anglebase - angleego
            escape_z = anglebase - angleenm

            return lock, escape, lock_xy, lock_z, escape_xy, escape_z, distance, delta_velocity

        for plane in self.plane_list:
            for plane_id in self.relationship[plane]['enemies']:
            #for plane_id in self.relationship[plane]['enemies'] + self.relationship[plane]['partners']:
                lock, escape, lock_xy, lock_z, escape_xy, escape_z, distance, delta_velocity = _get_relative_info(
                    plane, plane_id)
                self.state[plane][plane_id] = {}
                self.state[plane][plane_id]['lock'] = lock
                self.state[plane][plane_id]['escape'] = escape
                self.state[plane][plane_id]['lock_xy'] = lock_xy
                self.state[plane][plane_id]['lock_z'] = lock_z
                self.state[plane][plane_id]['escape_xy'] = escape_xy
                self.state[plane][plane_id]['escape_z'] = escape_z
                self.state[plane][plane_id]['distance'] = distance
                self.state[plane][plane_id]['delta_velocity'] = delta_velocity    #不做修改
        self.observation = self.state
        if self._check_dangerous_state()==False:
            return True
        else:
            return False


    def step(self, obs,actions):
        # 每一帧读取处理态势并返回指令的函数

        # 按格式创建命令字典。
        cmd_dict = {}

        # 预处理态势
        allies = obs.my_planes
        for ally in allies.values():
            ally.pos = Vec3(
                ally.x, ally.y, ally.z
            )
        enemies = obs.enemy_planes
        for enemy in enemies.values():
            enemy.pos = Vec3(
                enemy.x, enemy.y, enemy.z
            )

        awacs_infos = obs.awacs_infos
        rws_infos = obs.rws_infos

        # 对每架我方飞机计算行动

        for ally_ind, ally in allies.items():
            # 按格式设置每架飞机的动作、武器指令
            weapon_launch_info = {}

            action = [actions['aileron'],actions['elevator'], 0, actions['throttle']]
            # if np.random.rand() < 0.5:
            #     action = [-0.5 + 1 * self.rng.random(), -0.06 + 0.1*self.rng.random(), 0.01 * np.random.rand(), 0.5 + 0.5 * np.random.rand()]

            if obs.sim_time > 15:
                action = [0.3, 0.2, 0, 1.0]

            cmd_dict[ally_ind] = {
                'control': action
            }


            cmd_dict[ally_ind]['weapon'] = weapon_launch_info

            # print(action)
        # 返回指令。
        return cmd_dict

    def _check_dangerous_state(self):
        for plane in self.plane_list:
            if self.state[plane]['altitude']<1000 or self.state[plane]['altitude']>18000:
                print(f'{plane} has dangerous altitude {self.state[plane]["altitude"]}')
                return False
            # if self.state[plane]['alpha']<-25 or self.state[plane]['alpha']>25:
            #     print(f'{plane} has dangerous alpha {self.state[plane]["alpha"]}')
            #     return False
            # if self.state[plane]['beta']<-15 or self.state[plane]['beta']>15:
            #     print(f'{plane} has dangerous beta {self.state[plane]["beta"]}')
            #     return False
            if self.state[plane]['velocity'] < 0.2 or self.state[plane]['velocity'] > 1.8:
                print(f'{plane} has dangerous velocity {self.state[plane]["velocity"]}')
                return False
        return True

    def select_action(self, observation, target_info):
        # print("here")
        actions = {}
        if observation != None:
            self.observation = observation
        for plane in self.observation:
            # pdb.set_trace()
            if plane not in target_info.keys():
                continue
            if 'target_psi' in target_info[plane]:
                aileron, aileron_compensate = self.psi_control(plane, target_info[plane]['target_psi'],'psi')
            elif 'target_roll' in target_info[plane]:
                aileron, aileron_compensate = self.psi_control(plane, target_info[plane]['target_roll'],'roll')
            if 'target_height' in target_info[plane]:
                elevator = self.height_control(plane, target_info[plane]['target_height'], aileron_compensate,'height')
            elif 'target_pitch' in target_info[plane]:
                elevator = self.height_control(plane, target_info[plane]['target_pitch'], aileron_compensate, 'pitch')


            throttle = self.speed_control(plane, target_info[plane]['target_speed'])

            elevator = np.clip(elevator, self.action_last[plane]['elevator'] - 0.3, self.action_last[plane]['elevator'] + 0.3)

            aileron = np.clip(aileron, self.action_last[plane]['aileron'] - 0.3,self.action_last[plane]['aileron'] + 0.3)

            actions[plane] = {'aileron':aileron, 'elevator':elevator, 'throttle':throttle}

            self.action_last[plane] = {'aileron':aileron, 'elevator':elevator, 'throttle':throttle}
            # print(plane)
        # pdb.set_trace()
        return actions

    def height_control(self, plane, target, aileron_compensate, flag):
        elevator = 0

        #outer PID target_h to target_pitch
        #integral separated PID control
        if flag == 'height':
            current_height = self.observation[plane]['altitude']
            target_pitch = self.h_PID[plane].calculate(target, current_height)
        elif flag == 'pitch':
            target_pitch = target
        else:
            print(f'height control only support pitch or height')

        target_pitch += aileron_compensate+4

        if abs(self.state[plane]['roll'])<20:
            target_pitch = np.clip(target_pitch, -40, 40)
        else:
            target_pitch = np.clip(target_pitch, -15, 15)


        #inner PID target_pitch to elvator
        current_pitch = self.observation[plane]['pitch']
        elevator = self.pitch_PID[plane].calculate(target_pitch, current_pitch)

        elevator = np.clip(elevator, -1, 1)


        return elevator

    def speed_control(self, plane, target_speed):

        current_speed = self.observation[plane]['velocity']
        if current_speed>target_speed:
            throttle = 0
        else:
            throttle = 1

        return throttle


    def psi_control(self, plane, target, flag):

        #outer PID target_psi to target_roll
        #integral separated PID control
        if flag == 'psi':
            current_psi = self.observation[plane]['psi']
            # print("current psi",current_psi)
            # print("target psi",target)
            target_roll = self.psi_PID[plane].calculate(target, current_psi)
        elif flag == 'roll':
            target_roll = target
        else:
            print(f'height control only support psi or roll')

        if abs(self.state[plane]['pitch'])<80:
            target_roll = np.clip(target_roll, -80, 80)
        else:
            target_roll = np.clip(target_roll, -20, 20)

        aileron_compensate = self.K*abs(target_roll)
        #inner PID target_roll to aileron
        current_roll = self.observation[plane]['roll']
        aileron = self.roll_PID[plane].calculate(target_roll, current_roll)


        aileron = np.clip(aileron,-1, 1)


# =============================================================================
#         print(f'plane : {plane}')
#         print(f'current psi={current_psi}, target_psi={target_psi}')
#         print(f'current roll={current_roll}, target_roll={target_roll}')
# =============================================================================

        return aileron, aileron_compensate

    def blue_act(self, observation, target_info):
        actions = {}
        if observation != None:
            self.observation = observation
        for plane in self.observation:
            if plane == 1012:
                if 'target_psi' in target_info[plane]:
                    aileron, aileron_compensate = self.psi_control(plane, target_info[plane]['target_psi'],'psi')
                elif 'target_roll' in target_info[plane]:
                    aileron, aileron_compensate = self.psi_control(plane, target_info[plane]['target_roll'],'roll')
                if 'target_height' in target_info[plane]:
                    elevator = self.height_control(plane, target_info[plane]['target_height'], aileron_compensate,'height')
                elif 'target_pitch' in target_info[plane]:
                    elevator = self.height_control(plane, target_info[plane]['target_pitch'], aileron_compensate, 'pitch')


                throttle = self.speed_control(plane, target_info[plane]['target_speed'])

                elevator = np.clip(elevator, self.action_last[plane]['elevator'] - 0.3, self.action_last[plane]['elevator'] + 0.3)

                aileron = np.clip(aileron, self.action_last[plane]['aileron'] - 0.3,self.action_last[plane]['aileron'] + 0.3)

                actions[plane] = {'aileron':aileron, 'elevator':elevator, 'throttle':throttle}

                self.action_last[plane] = {'aileron':aileron, 'elevator':elevator, 'throttle':throttle}

        return actions
    def red_act(self,obs):### 传进来的obs_shape   [agent_num,obs_dim]
        agent_num = len(obs)
        obs_dim = len(obs[0])
        trainer = MAPPO(agent_num,obs_dim,action_dim)

class Missile():
    def __init__(self, target, init_state, name):
        self.target = target
        self.longitude = init_state['longitude']
        self.latitude = init_state['latitude']
        self.altitude = init_state['altitude']
        self.name = name
        self.T = 0.05
        self.dis_T = 2 * 340 * 0.2/1000
        self.init_flag = 1

        self.hit = 0

    def update(self, obs):
        target_lon = obs[self.target]['longitude']
        target_lat = obs[self.target]['latitude']
        target_alt = obs[self.target]['altitude']
        delta_longitude = abs(target_lon - self.longitude)
        delta_latitude = abs(target_lat - self.latitude)
        delta_altitude = abs(target_alt - self.altitude)
        middle_latitude = (target_lat - self.latitude) / 2
        dis = ((delta_longitude * 111194 * math.cos(middle_latitude)) ** 2 + (delta_latitude * 111194) ** 2 + (
            delta_altitude) ** 2) ** 0.5/1000
        if dis < 0.2:
            self.hit = 1
        percent = self.dis_T / dis
        new_lon = target_lon * percent + self.longitude * (1 - percent)
        new_lat = target_lat * percent + self.latitude * (1 - percent)
        new_alt = target_alt * percent + self.altitude * (1 - percent)

        self.longitude = new_lon
        self.latitude = new_lat
        self.altitude = new_alt

        return new_lon, new_lat, new_alt

    def get_tacview(self):
        if self.init_flag == 1:
            self.init_flag = 0
            return self.name + ',T=' + str(self.longitude) + '|' + str(self.latitude) + '|' + str(
                self.altitude) + ',Type=Weapon+Missile,Color=Blue,Name=AIM_120\n'
        else:
            return self.name + ',T=' + str(self.longitude) + '|' + str(self.latitude) + '|' + str(
                self.altitude) + '\n'

if __name__=="__main__":
    test = Approach()
    init_info = {'time':'0'}
    init_info.update(test.trajectory)
    obs = test.get_obs()
    print('start !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    for i in range(3):
        for i in range(500):
            target_info = {'101': {'target_height': 6000, 'target_speed': 0.7, 'target_psi': 0}}
            actions = test.select_action(target_info)
            obs = test.step(actions)

        for i in range(500):
            target_info = {'101': {'target_height': 6000, 'target_speed': 0.7, 'target_psi': 120}}
            actions = test.select_action(target_info)
            obs = test.step(actions)
        for i in range(500):
            target_info = {'101': {'target_height': 6000, 'target_speed': 0.7, 'target_psi': -120}}
            actions = test.select_action(target_info)
            obs = test.step(actions)
    test.export()




