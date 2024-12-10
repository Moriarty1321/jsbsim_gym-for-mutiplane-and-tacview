import pdb

import gym
from gym import spaces
import jsbsim
import numpy as np
import gym
from vanillaDQN.dqn import DQNAgent
from common.utils import mini_batch_train



############################################ 1.环境类 ###########################################
a = 100
class MultiAircraftEnv(gym.Env):
    def __init__(self, plane_id = ["101","201"],num_aircrafts=2):
        super(MultiAircraftEnv, self).__init__()

        # 定义飞机的数量
        self.plane_id = plane_id
        self.num_aircrafts = num_aircrafts
        self._create_records = False
        self.uid = "A0100"
        self.model = "f16"
        self.color = "Red"
        # 创建多个 JSBSim 实例 创建为字典格式，飞机id对应一个实体
        self.fdms = {}
        for key in self.plane_id:
            self.fdms[key] = jsbsim.FGFDMExec(".")
        self.current_step = 0
        self.down_sample = 4
        # pdb.set_trace()
        # 定义每架飞机的初始条件
        self.initial_conditions = {
            "101":
            {
                'latitude': 37.618889,  # 纬度
                'longitude': -122,  # 经度
                'altitude': 20000,  # 高度
                'airspeed': 1,  # 马赫
                'heading': -90,  # 航向
            },
            "201":
            {
                'latitude': 37.620000,
                'longitude': -122.1,
                'altitude': 21000,
                'airspeed': 1,
                'heading': 90,
            }
        }

        # 加载飞机模型和飞行脚本
        for id, fdm in self.fdms.items():
            fdm.load_model('f16')
            # 设置初始条件
            fdm['ic/lat-gc-deg'] = self.initial_conditions[id]['latitude']
            fdm['ic/long-gc-deg'] = self.initial_conditions[id]['longitude']
            fdm['ic/h-sl-ft'] = self.initial_conditions[id]['altitude']
            fdm['ic/mach'] = self.initial_conditions[id]['airspeed']
            fdm['ic/psi-true-deg'] = self.initial_conditions[id]['heading']  # 设置航向
            fdm['propulsion/engine/set-running'] = 1
            # fdm.set_dt(1/20)
            fdm.run_ic()  # 初始化模拟
            # pdb.set_trace()

        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11 * num_aircrafts,), dtype=np.float32)

    def step(self, actions):
        # 确保动作的数量与飞机数量一致
        self.current_step += 1
        # assert len(actions) == self.num_aircrafts, "Number of actions must match number of aircrafts"
        # pdb.set_trace()
        # 运行模拟并设置控制输入
        for id, fdm in self.fdms.items():
            if id in actions.keys():
                # fdm['fcs/throttle-cmd-norm'] = (actions[i][0] + 1) / 2
                fdm['fcs/throttle-cmd-norm'] = actions[id]["throttle"]
                # fdm.set_property_value("fcs/throttle-cmd-norm", 1)
                fdm['fcs/elevator-cmd-norm'] = actions[id]["elevator"]
                fdm['fcs/aileron-cmd-norm'] = actions[id]["aileron"]
            # fdm['fcs/rudder-cmd-norm'] = actions[id][3]

            # for _ in range(self.down_sample):
            #     # Freeze fuel consumption
            #     fdm.set_property_value("propulsion/tank/contents-lbs", 1000)
            #     fdm.set_property_value("propulsion/tank[1]/contents-lbs", 1000)
            #
            #     # Set gear up
            #     fdm.set_property_value("gear/gear-cmd-norm", 0.0)
            #     fdm.set_property_value("gear/gear-pos-norm", 0.0)

                fdm.run()


            # 获取观察值
        observations = {}
        for id,fdm in self.fdms.items():
            latitude = fdm['position/lat-gc-deg']
            longitude = fdm['position/long-gc-deg']
            altitude = fdm['position/h-sl-ft']/3.28
            mach = fdm['velocities/mach']
            heading = fdm['attitude/psi-deg']
            roll = fdm['attitude/phi-deg']
            pitch = fdm['attitude/theta-deg']
            yaw = fdm['attitude/psi-deg']
            u = fdm['velocities/u-fps'] * 0.3048  # 转换为米/秒
            v = fdm['velocities/v-fps'] * 0.3048  # 转换为米/秒
            w = fdm['velocities/w-fps'] * 0.3048  # 转换为米/秒
            # observations[id] = [latitude, longitude, altitude, mach, heading, roll, pitch, yaw]
            observations[id] = {
                "lat": latitude,
                "lon": longitude,
                "height": altitude,
                "mach": mach,
                "psi": heading,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "v_x": u,
                "v_y": v,
                "v_z": w,
            }
            # print(
            #     f"Altitude: {altitude:.2f} ft, Airspeed: {mach:.2f} mach, Heading: {heading:.2f} deg")
            # print(f"Roll: {roll:.2f} deg, Pitch: {pitch:.2f} deg, Yaw: {yaw:.2f} deg")
        # print(observations)


        # 奖励在此处设置
        rewards = [0.0] * self.num_aircrafts  # 每个飞机的奖励

        # 终止条件在此处设置
        dones = [False] * self.num_aircrafts

        # 返回结果
        # pdb.set_trace()
        return observations, 0, 0, 0

    def reset(self):
        # 重置每架飞机的初始条件
        self.current_step = 0
        for id, fdm in self.fdms.items():
            fdm['ic/lat-gc-deg'] = self.initial_conditions[id]['latitude']
            fdm['ic/long-gc-deg'] = self.initial_conditions[id]['longitude']
            fdm['ic/h-sl-ft'] = self.initial_conditions[id]['altitude']
            fdm['ic/mach'] = self.initial_conditions[id]['airspeed']
            fdm['ic/psi-true-deg'] = self.initial_conditions[id]['heading']  # 设置航向
            fdm.run_ic()  # 初始化模拟

        # 获取初始观察值
        observations = {}
        for id,fdm in self.fdms.items():
            latitude = fdm['position/lat-gc-deg']
            longitude = fdm['position/long-gc-deg']
            altitude = fdm['position/h-sl-ft']/3.28
            mach = fdm['velocities/mach']
            heading = fdm['attitude/psi-deg']
            roll = fdm['attitude/phi-deg']
            pitch = fdm['attitude/theta-deg']
            yaw = fdm['attitude/psi-deg']
            u = fdm['velocities/u-fps'] * 0.3048  # 转换为米/秒
            v = fdm['velocities/v-fps'] * 0.3048  # 转换为米/秒
            w = fdm['velocities/w-fps'] * 0.3048  # 转换为米/秒
            # observations[id] = [latitude, longitude, altitude, mach, heading, roll, pitch, yaw]
            observations[id] = {
                "lat": latitude,
                "lon": longitude,
                "height": altitude,
                "mach": mach,
                "psi": heading,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "v_x": u,
                "v_y": v,
                "v_z": w,
            }

        return observations


    def render_tacview(self, mode="txt", filepath='./JSBSimRecording.txt.acmi'):
        if mode == "txt":
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * 0.2  # 记录的时间戳
                f.write(f"#{timestamp:.2f}\n")
                i = 0
                for id,sim in self.fdms.items():
                    latitude = sim['position/lat-gc-deg']
                    longitude = sim['position/long-gc-deg']
                    altitude = sim['position/h-sl-ft']
                    roll = sim['attitude/phi-deg']
                    pitch = sim['attitude/theta-deg']
                    yaw = sim['attitude/psi-deg']
                    airspeed = sim['velocities/vc-kts']

                    log_msg = id + f",T={longitude}|{latitude}|{altitude}|{roll}|{pitch}|{yaw},V={airspeed},Name=f16,Color=Red"
                    f.write(log_msg + "\n")
                    i += 1

        else:
            raise NotImplementedError






############################## 2.下层智能体类 #################################################
class avoid_missile_agent:
    def __init__(self):
        return 0
    def take_action(self, missile_flag, missile_cnt):
        return 0


class rush_agent:
    def __init__(self):
        return 0
    def take_action(self, missile_flag, missile_cnt):
        return 0

class dog_fight_agent:
    def __init__(self):
        return 0
    def take_action(self, missile_flag, missile_cnt):
        return 0

class Blue_agent:
    def __init__(self):
        return 0
    def take_action(self, missile_flag, missile_cnt):
        return 0




################################ 3.dqn智能体在环境中交互 #############################################
if __name__ == '__main__':
    gym.envs.register(
        id='MultiAircraft-v0',
        entry_point='__main__:MultiAircraftEnv',
        max_episode_steps=20000,
    )
    env = gym.make('MultiAircraft-v0')

    MAX_EPISODES = 100
    MAX_STEPS = 5000
    BATCH_SIZE = 100

    agent = DQNAgent(env, use_conv=False)
    episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)



