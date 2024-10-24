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

i = 0
while not done:
    action = [[1,1,1],[1,-1,1],[1,0,0],[1,1,1]]
    obs, rew, done, _ = env.step(action)
    print(i)
    i += 1
    env.render_tacview()
env.close()
```
and in jsbsim_gym.py class JSBSimEnv(gym.Env), we can load multiple planes  
```
for i in range(self.num_agent):

    plane_list.append(jsbsim.FGFDMExec(root, None))
    plane_list[i].set_debug_level(0)

    # Load F-16 model and set initial conditions
    plane_list[i].load_model('f16')
    self._set_initial_conditions(plane_list[i], v, height)
    plane_list[i].run_ic()
    height += 500

self.simulation = plane_list
```
every single jsbsim.FGFDMExec(root,None) is a plane class, I put these planes in a list self.simulation for using, you can set initial velocity and height by using self._set_initial_conditions  

## render
you can use following code to render your plane by .acmi file through tacview.  
```
env.render_tacview()
```

