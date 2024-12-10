import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import math
import gym


from air_combat.blue_rule_hd_original import Rule_Agent
### 导入智能体类
from air_combat.lower_agents import rush_agent, avoid_missile_agent, dog_fight_agent, Blue_agent, Missile, BaseAgent
from air_combat.tacview import TacviewRecorder


def mini_batch_train(env, upper_agent, max_episodes, max_steps, batch_size):
    #######   训练全流程
    #######   上层dqn网络输入状态  输出 0 1 2
    #######   下层接受0 1 2 选择相应智能体来对飞机进行策略输出
    ########  下层规则智能体由rule agent实现，需要对其初始化

    #####   规则智能体初始化
    episode_rewards = []
    plane_dict = {"101":'New0111_red',"201":'New0111_red'}
    relationship = {"101":{"enemies":["201"]},"201":{"enemies":["101"]}}
    rule_agent = Rule_Agent(plane_dict,relationship)

    ####   record初始化
    recorder_settings = {
        'Env':
            {'FileType': 'text/acmi/tacview',
             'FileVersion': '2.1',
             'ReferenceTime': '2022-07-06T08:39:21.153Z',
             'DeltaTime': '0.05'},
        'Plane':
            {
                '101': {'Type': 'Air+FixedWing', 'Model': 'F-16C', 'Color': 'Red'},
                '201': {'Type': 'Air+FixedWing', 'Model': 'F-16C', 'Color': 'Blue'}
                 },
        'RealTime': False,
        'Info': {'IP': 'localhost', 'port': 7788}
    }
    recorder = TacviewRecorder(recorder_settings)
    init_info = {'time': '0'}
    init_info.update(rule_agent.trajectory)
    recorder.reset(init_info)



    for episode in range(max_episodes):

        ##### missile 初始化
        target_info = {}
        missile_list = []
        missile_flag = [1] * 2
        missile_cnt = 0
        raw_obs = env.reset()  ###   字典
        # print(raw_obs)
        state =  [value for subdict in raw_obs.values() for value in subdict.values()]  ###   列表
        # pdb.set_trace()
        episode_reward = 0
        rule_agent._parse_observation(raw_obs)

        ### recorder初始化
        #recorder = TacviewRecorder(recorder_settings)
        init_info = {'time': '0'}
        init_info.update(rule_agent.trajectory)
        recorder.reset(init_info)
        # pdb.set_trace()
        for step in range(max_steps): #### 每秒决策20步
            #### 上层10s决策一次
            if step % 200 == 0:
                upper_action = upper_agent.get_action(state)###   上层动作0 1 2
                print("当前选择智能体",upper_action)

                ###  下层选择智能体
                if upper_action == 0:
                    # lower_agent = avoid_missile_agent(rule_agent)
                    #### 调用模板智能体
                    lower_agent = BaseAgent(env,rule_agent)
                elif upper_action == 1:
                    # lower_agent = rush_agent(rule_agent)
                    lower_agent = BaseAgent(env, rule_agent)
                elif upper_action == 2:
                    # lower_agent = dog_fight_agent(rule_agent)
                    lower_agent = BaseAgent(env, rule_agent)

            ### 红方智能体动作选择
            lower_action,next_raw_state, reward, done = lower_agent.step(state)   ########规则智能体用rule agent的话需要传入字典，神经网络传入ndarray
            env.step(lower_action)

            #### 蓝方智能体动作选择
            blue_agent = Blue_agent(rule_agent)
            blue_info = blue_agent.take_action(missile_flag,missile_cnt)
            blue_action = rule_agent.select_action(None,blue_info)
            env.step(blue_action)



            # pdb.set_trace()
            rule_agent._parse_observation(next_raw_state)
            next_state = [value for subdict in next_raw_state.values() for value in subdict.values()] ### 字典转列表
            upper_agent.replay_buffer.push(state, upper_action, reward, next_state, done)

            ### 发弹
            if missile_flag[0]==1:
                missile_list.append(Missile('101', raw_obs["201"], '301'))
                missile_flag[0] = 0
                missile_interval = 20
            elif missile_flag[1]==1:
                if missile_interval <=0:
                    missile_list.append(Missile('101', raw_obs["201"], '302'))
                    missile_flag[1] = 0
                else:
                    missile_interval-=1
            for missile in missile_list:
                missile.update(raw_obs)
                missile_info = missile.get_tacview()
                recorder.insert(missile_info)
            for ind, missile in enumerate(missile_list):
                if missile.hit == 1:
                    missile_list.pop(ind)
                    recorder.insert('-' + missile.name + '\n')
                    recorder.insert('-' + missile.target + '\n')
                elif missile.hit == 2:
                    missile_list.pop(ind)
                    recorder.insert('-' + missile.name + '\n')
                else:
                    missile.remain_time -= 1

            ### 可视化
            recorder.record(rule_agent.trajectory)

            ### 奖励函数在此处设计，



            episode_reward += reward

            ### 上层dqn根据batch更新
            if len(upper_agent.replay_buffer) > batch_size:
                upper_agent.update(batch_size)
            ### 下层智能体更新
            # lower_agent.update(batch_size)

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state
        recorder.export('./temp.txt.acmi')
    return episode_rewards

