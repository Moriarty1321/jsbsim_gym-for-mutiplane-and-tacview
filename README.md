# jsbsim_gym-for-mutiplane-and-tacview
this project support load multi plane for mutiagent RL and can be rendered using tacview

## environment  
you can use environment.yml or requirements.txt to install conda virtual environment.  
```
conda install --yes --file requirements.txt  
```
or  
```
conda env create -f environment.yml
```
## usage  
this simulator packs jsbsim by format of gym, also can load multiple planes and control them by action[aileron, elevator, rudder, throttle]  
rudder is usually set to 0.  
you can use following command to simply use this simulator  
```
python train.py
```
in train.py here I set an action to control three planes, if you want to make planes flying stably, you can use RL algorithms or other control algorithms to train or control planes.  
```
env = gym.make("JSBSim-v0")
obs = env.reset()
done = False
log_path = path.join(path.abspath(path.dirname(__file__)), 'logs')

# import pdb
# pdb.set_trace()
i = 0
while not done:
    action = [[1,1,1],[1,-1,1],[1,0,0],[1,1,1]]
    obs, rew, done, _ = env.step(action)
    print(i)
    i += 1
    env.render_tacview()
env.close()
```
