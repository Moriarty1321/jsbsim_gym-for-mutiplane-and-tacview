# jsbsim_gym-for-hier-RL-and-tacview用于强化学习以及tacview可视化的jsbism_gym  
项目特点：  
1.将jsbsim包装成gym格式方便与强化学习算法对接，支持多架飞机，支持用tacview可视化  
2.该项目是一个业务逻辑上的分层框架，上层智能调用下层智能体，目前下层为规则智能体，上层为dqn算法，根据当前态势选择调用下层三个规则智能体的一个，当然你也可以利用该环境训练自己的下层智能体  
3.将下层agent从环境中解耦，该项目应用场景为，下层预训练智能体由不同的人提供，共用一个相同的环境，为方便开发和维护，将智能体及奖励从环境中解耦成一个BaseAgent Class，这样每个人只需要在按照模板设计自己的类，而不需要共同维护一个大的环境。  
------------------------------------------------------  
在air_combat文件夹下lower_agents.py里定义了三种规则智能体以及兼容规则和预训练智能体的BaseAgent Class,你可以在该模板里定义你自己的规则或者预训练智能体，并根据self.rule来进行切换，并且奖励函数从环境中解耦出来，在BaseAgent.step里定义，即面对其他人的环境以及涉及多人合作的多智能体场景中，你只需要关注你自己的预训练智能体，而不需要多个人对一个环境类进行编写。    
env.py里是gym形式的jsbsim的类，你可以在test_main里根据自己的需求做测试，目前test_main里是我自己分层架构的代码流程  
使用方法和gym一样  
```
gym.envs.register(  
        id='MultiAircraft-v0',  
        entry_point='__main__:MultiAircraftEnv',  
        max_episode_steps=20000,  
    )  
env = gym.make('MultiAircraft-v0')  
```  
this project support load multi plane for mutiagent RL and can be rendered using tacview， still updating

## todo list
-1 use a config json to create multiple plane and initial conditions（already done）  
-2 use rl algorithm to train an agent to fly stably  
-3 add missle module  
-4 apply hierachical rl structure using this simulator


## environment  
you can use environment.yml or requirements.txt to install conda virtual environment.  
```
conda install --yes --file requirements.txt  
```
or  
```
conda env create -f environment.yml
```

every single jsbsim.FGFDMExec(root,None) is a plane class, I put these planes in a list self.simulation for using, you can set initial velocity, height or other conditions by using self._set_initial_conditions  



