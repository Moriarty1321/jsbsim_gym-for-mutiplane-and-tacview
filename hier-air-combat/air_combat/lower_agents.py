import math
import pdb
from collections import deque
from math import degrees as r2d
from math import radians as d2r
import numpy as np

import torch.nn as nn
from abc import ABC
import sys
import os
# Deal with import error
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Literal

from model.baseline_actor import BaselineActor, BasicBuffer
import torch.optim as optim
from env import MultiAircraftEnv


def get_root_dir():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')

def in_range_deg(angle):
    """ Given an angle in degrees, normalises in (-180, 180] """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def in_range_rad(angle):
    """ Given an angle in rads, normalises in (-pi, pi] """
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle

################# 定义导弹以及下层智能体
############ 规则智能体#########################################################
class Missile():
    def __init__(self, target, init_state, name):
        self.target = target
        # pdb.set_trace()
        self.longitude = init_state['lon']
        self.latitude = init_state['lat']
        self.altitude = init_state['height']
        self.name = name
        self.T = 0.2
        self.dis_T = 2.5 * 340 * 0.2
        self.init_flag = 1
        self.remain_time = 20
        self.hit = 0

    def update(self, obs):
        target_lon = obs[self.target]['lon']
        target_lat = obs[self.target]['lat']
        target_alt = obs[self.target]['height']
        delta_longitude = abs(target_lon - self.longitude)
        delta_latitude = abs(target_lat - self.latitude)
        delta_altitude = abs(target_alt - self.altitude)
        middle_latitude = (target_lat - self.latitude) / 2
        dis = ((delta_longitude * 111194 * math.cos(middle_latitude)) ** 2 + (delta_latitude * 111194) ** 2 + (
            delta_altitude) ** 2) ** 0.5
        if dis < self.dis_T:
            if self.remain_time >=0:
                self.hit = 1
            else:
                self.hit = 2
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
            return self.name + ',T=' + str(self.longitude) + '|' + str(self.latitude) + '|' + str(self.altitude) + '\n'




class avoid_missile_agent:
    def __init__(self,agent):
        self.target_info = {}
        self.agents = agent

    def take_action(self, missile_flag, missile_cnt):

        if missile_flag[0] == 0 or missile_flag[1] == 0:
            self.target_info["101"] = {"target_height":7000, "target_speed":1.0, "target_psi":90+self.agents.state["101"]["psi"]}  #### psi解析有问题

        else:
            self.target_info["101"] = {"target_height": 7000, "target_speed": 1.0, "target_psi": 0}
        return self.target_info


class rush_agent:
    def __init__(self,agent):
        self.target_info = {}
        self.agents = agent
    def take_action(self, missile_flag, missile_cnt):
        self.target_info["101"] = {"target_height": self.agents.state["201"]["altitude"], "target_speed": 1.0,
                                   "target_psi": self.agents.state["101"]["psi"]+self.agents.state["101"]["201"]["lock_xy"]}
        return self.target_info

class dog_fight_agent:
    def __init__(self,agent):
        self.target_info = {}
        self.agents = agent
    def take_action(self, missile_flag, missile_cnt):

        if missile_flag[0] == 0 or missile_flag[1] == 0:
            self.target_info["101"] = {"target_height":7000, "target_speed":1.0, "target_psi":-90}  #### psi解析有问题
        else:
            self.target_info["101"] = {"target_height": 7000, "target_speed": 1.0, "target_psi": 0}
        if missile_cnt == 2:
            self.target_info["101"] = {"target_height": 7000, "target_speed": 1.0,
                                       "target_psi": self.agents.state["101"]["psi"] +self.agents.state["101"]["201"]["lock_xy"]}
        return self.target_info

class Blue_agent:
    def __init__(self,agent):
        self.target_info = {}
        self.agents = agent
    def take_action(self, missile_flag, missile_cnt):
        self.target_info["201"] = {"target_height": 7000, "target_speed": 1.0,
                                   "target_psi": self.agents.state["201"]["psi"] + r2d(self.agents.state["201"]["101"][
                                       "lock_xy"])}
        return self.target_info

####################### 强化学习智能体模板 ####################################

class BaseAgent(nn.Module):
    def __init__(self, env, agent, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.rule = True
        if not self.rule:
            self.model_path = 'baseline_model.pt'
            self.actor = BaselineActor()
            self.actor.load_state_dict(torch.load(self.model_path, weights_only=False, map_location=torch.device('cpu')))
            self.actor.eval()
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.replay_buffer = BasicBuffer(max_size=buffer_size)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.optimizer = torch.optim.Adam(self.actor.parameters())
        else:
            self.env = env
            self.agents = agent
            self.target_info = {}


    def step(self, state):
        if not self.rule:
            state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
            lower_action = self.model.forward(state)
            next_raw_state, reward, done, _ = self.env.step(lower_action)
        else:
            self.target_info["101"] = {"target_height": 7000, "target_speed": 1.0,"target_psi": 90}
            lower_action = self.agents.select_action(None,self.target_info)
            next_raw_state, reward, done, _ = self.env.step(lower_action)

        ### 奖励函数在此设计

        ### 主循环里的step可以不需要了，在这里返回
        return lower_action, next_raw_state, reward, done

    def compute_loss(self, batch):
        #### loss 在此处自定义
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load_model(self,model_path):
        self.actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))



################################################## 下边没啥用###################################################
class BaselineAgent(ABC):
    def __init__(self, agent_id) -> None:
        self.model_path = 'baseline_model.pt'
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path, weights_only=False,map_location=torch.device('cpu')))
        self.actor.eval()
        self.agent_id = agent_id
        self.reset()
    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    # @abstractmethod
    # def set_delta_value(self, env, task):
    #     raise NotImplementedError

    def get_observation(self, raw_obs):

        norm_obs = np.zeros(12)
        norm_obs[0] = (raw_obs["101"]["height"]-raw_obs["201"]["height"]) / 1000          #  0. ego delta altitude  (unit: 1km)

        ### delta heading 处理
        ego_v = np.linalg.norm([raw_obs["101"]["v_x"], raw_obs["101"]["v_y"]])   ###### 速度是m/s
        delta_x, delta_y = raw_obs["201"]["lat"] - raw_obs["101"]["lat"], raw_obs["201"]["lon"] - raw_obs["101"]["lon"]
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * raw_obs["101"]["v_x"] + delta_y * raw_obs["101"]["v_y"]
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([raw_obs["101"]["v_x"], raw_obs["101"]["v_y"]], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag   ##### rad
        # print("delta_heading", delta_heading)

        norm_obs[1] = in_range_rad(delta_heading)   #  1. ego delta heading   (unit rad)
        norm_obs[2] = raw_obs["101"]["mach"]-raw_obs["201"]["mach"]         #  2. ego delta velocities_u  (unit: mh)
        norm_obs[3] = raw_obs["101"]["height"] / 5000                  #  3. ego_altitude (unit: km)
        norm_obs[4] = np.sin(in_range_rad(raw_obs["101"]["roll"]))                 #  4. ego_roll_sin
        norm_obs[5] = np.cos(in_range_rad(raw_obs["101"]["roll"]))                 #  5. ego_roll_cos
        norm_obs[6] = np.sin(in_range_rad(raw_obs["101"]["pitch"]))                 #  6. ego_pitch_sin
        norm_obs[7] = np.cos(in_range_rad(raw_obs["101"]["pitch"]))                 #  7. ego_pitch_cos
        norm_obs[8] = raw_obs["101"]["v_x"] / 340                   #  8. ego_v_x   (unit: mh)
        norm_obs[9] = raw_obs["101"]["v_y"] / 340                   #  9. ego_v_y    (unit: mh)
        norm_obs[10] = raw_obs["101"]["v_z"] / 340                  #  10. ego_v_z    (unit: mh)
        norm_obs[11] = raw_obs["101"]["mach"]                #  11. ego_vc        (unit: mh)
        norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
        print(norm_obs)
        return norm_obs

    def get_action(self, raw_obs):
        # delta_value = self.set_delta_value(env, task)
        observation = self.get_observation(raw_obs)
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return action


class PursueAgent(BaselineAgent):
    def __init__(self, agent_id) -> None:
        super().__init__(agent_id)

    def set_delta_value(self, env, task):
        # NOTE: only adapt for 1v1
        ego_uid, enm_uid = list(env.agents.keys())[self.agent_id], list(env.agents.keys())[(self.agent_id+1)%2]
        ego_x, ego_y, ego_z = env.agents[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.agents[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.agents[enm_uid].get_position()
        # delta altitude
        delta_altitude = enm_z - ego_z
        # delta heading
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        # delta velocity
        delta_velocity = env.agents[enm_uid].get_property_value(c.velocities_u_mps) - \
                         env.agents[ego_uid].get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])


class ManeuverAgent(BaselineAgent):
    def __init__(self, agent_id, maneuver: Literal['l', 'r', 'n']) -> None:
        super().__init__(agent_id)
        self.turn_interval = 30
        self.dodge_missile = False # if set true, start turn when missile is detected
        if maneuver == 'l':
            self.target_heading_list = [0]
        elif maneuver == 'r':
            self.target_heading_list = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]
        elif maneuver == 'n':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
        elif maneuver == 'triangle':
            self.target_heading_list = [np.pi/3, np.pi, -np.pi/3]*2
        self.target_altitude_list = [6000] * 6
        self.target_velocity_list = [243]  * 6

    def reset(self):
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))
        self.init_heading = None

    def set_delta_value(self, env, task):
        step_list = np.arange(1, len(self.target_heading_list)+1) * self.turn_interval / env.time_interval
        uid = list(env.agents.keys())[self.agent_id]
        cur_heading = env.agents[uid].get_property_value(c.attitude_heading_true_rad)
        if self.init_heading is None:
            self.init_heading = cur_heading
        if not self.dodge_missile or task._check_missile_warning(env, self.agent_id) is not None:
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            delta_heading = self.init_heading + self.target_heading_list[i] - cur_heading
            delta_altitude = self.target_altitude_list[i] - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = self.target_velocity_list[i] - env.agents[uid].get_property_value(c.velocities_u_mps)
            self.step += 1
        else:
            delta_heading = self.init_heading  - cur_heading
            delta_altitude = 6000 - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = 243 - env.agents[uid].get_property_value(c.velocities_u_mps)

        return np.array([delta_altitude, delta_heading, delta_velocity])


if __name__ == '__main__':
    from gym import spaces
    import jsbsim
    import numpy as np
    import gym

    # obs = {'101': {'lat': 37.6188889999999, 'lon': -122.00002987903054, 'height': 6097.560975625657, 'mach': 0.999827266659073, 'psi': 269.9999816848979, 'roll': -3.6814220918286e-12, 'pitch': 2.3607486186248605e-05, 'yaw': 269.9999816848979, 'v_x': 316.0009771117659, 'v_y': 0.00023512665156900364, 'v_z': 0.0016368824946044083}, '201': {'lat': 37.619999999982404, 'lon': -122.09997024102562, 'height': 6402.439026925259, 'mach': 0.9998337367193172, 'psi': 90.00001824196372, 'roll': -1.3952553642437593e-11, 'pitch': 2.351227807349746e-05, 'yaw': 90.00001824196372, 'v_x': 314.7436201468973, 'v_y': 0.00023483178137327746, 'v_z': 0.0043060581897813345}}
    # obs = {'101': {'lat': 37.6188889999999, 'lon': -122.00002987903054, 'height': 6097.560975625657,
    #                'mach': 0.999827266659073, 'psi': 269.9999816848979, 'roll': -3.6814220918286e-12,
    #                'pitch': 2.3607486186248605e-05, 'yaw': 269.9999816848979, 'v_x': 316.0009771117659,
    #                'v_y': 0.00023512665156900364, 'v_z': 0.0016368824946044083},
    #        '201': {'lat': 37.619999999982404, 'lon': -122.09997024102562, 'height': 6402.439026925259,
    #                'mach': 0.9998337367193172, 'psi': 90.00001824196372, 'roll': -1.3952553642437593e-11,
    #                'pitch': 2.351227807349746e-05, 'yaw': 90.00001824196372, 'v_x': 314.7436201468973,
    #                'v_y': 0.00023483178137327746, 'v_z': 0.0043060581897813345}}
    gym.envs.register(
        id='MultiAircraft-v0',
        entry_point='__main__:MultiAircraftEnv',
        max_episode_steps=20000,
    )
    env = gym.make('MultiAircraft-v0')
    obs = env.reset()
    agent = BaselineAgent("101")
    action = agent.get_action(obs)  ### array([ 7,  1, 40, 28], dtype=int64)
    pdb.set_trace()
    action_dict = {}
    for i in range(5000):
        action = agent.get_action(obs)
        action_dict = {}
        action_dict["101"] = {"throttle":action[0], "aileron":action[1], "elevator":action[2]}
        obs, reward, done, info = env.step(action)
    print(action)

