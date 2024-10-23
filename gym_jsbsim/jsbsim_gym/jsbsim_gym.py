import jsbsim
import gym

import numpy as np
import sys
print(sys.path)
# sys.path.append("/")
from .visualization.rendering import Viewer, load_mesh, load_shader, RenderObject, Grid
from .visualization.quaternion import Quaternion
import pdb
# Initialize format for the environment state vector
a = 100
STATE_FORMAT = [
    "position/lat-gc-rad",
    "position/long-gc-rad",
    "position/h-sl-meters",
    "velocities/mach",
    "aero/alpha-rad",
    "aero/beta-rad",
    "velocities/p-rad_sec",
    "velocities/q-rad_sec",
    "velocities/r-rad_sec",
    "attitude/phi-rad",
    "attitude/theta-rad",
    "attitude/psi-rad",
]

STATE_LOW = np.array([
    -np.inf,
    -np.inf,
    0,
    0,
    -np.pi,
    -np.pi,
    -np.inf,
    -np.inf,
    -np.inf,
    -np.pi,
    -np.pi,
    -np.pi,
    -np.inf,
    -np.inf,
    0,
])

STATE_HIGH = np.array([
    np.inf,
    np.inf,
    np.inf,
    np.inf,
    np.pi,
    np.pi,
    np.inf,
    np.inf,
    np.inf,
    np.pi,
    np.pi,
    np.pi,
    np.inf,
    np.inf,
    np.inf,
])

# Radius of the earth
RADIUS = 6.3781e6

class JSBSimEnv(gym.Env):
    def __init__(self, root='.'):
        super().__init__()
        self.num_agent = 3
        # Set observation and action space format
        self.observation_space = gym.spaces.Box(STATE_LOW, STATE_HIGH, (15,))
        self.action_space = gym.spaces.Box(np.array([-1,-1,-1,0]), 1, (4,))
        self._create_records = False
        self.uid = "A0100"
        self.model = "f16"
        self.color = "Red"
        ### 每个simulation是一架飞机
        # Initialize JSBSim
        plane_list = []
        v = 900
        height = 4000
        for i in range(self.num_agent):

            plane_list.append(jsbsim.FGFDMExec(root, None))
            plane_list[i].set_debug_level(0)

            # Load F-16 model and set initial conditions
            plane_list[i].load_model('f16')
            self._set_initial_conditions(plane_list[i], v, height)
            plane_list[i].run_ic()
            height += 500

        self.simulation = plane_list

        # pdb.set_trace()





        self.down_sample = 4
        self.state = np.zeros(12)
        self.goal = np.zeros(3)
        self.dg = 100
        self.viewer = None

        ### for render tacview
        self.current_step = 0

    def _set_initial_conditions(self, aircraft, v, h):
        # Set engines running, forward velocity, and altitude
        aircraft.set_property_value('propulsion/set-running', -1)
        aircraft.set_property_value('ic/u-fps', v)
        aircraft.set_property_value('ic/h-sl-ft', h)
    
    def step(self, action):
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action

        self.current_step += 1

        # Pass control inputs to JSBSim   each plane
        for i in range(self.num_agent):
            self.simulation[i].set_property_value("fcs/aileron-cmd-norm", roll_cmd[i])
            self.simulation[i].set_property_value("fcs/elevator-cmd-norm", pitch_cmd[i])
            self.simulation[i].set_property_value("fcs/rudder-cmd-norm", yaw_cmd[i])
            self.simulation[i].set_property_value("fcs/throttle-cmd-norm", throttle[i])

        # We take multiple steps of the simulation per step of the environment
        for _ in range(self.down_sample):
            # Freeze fuel consumption
            for i in range(self.num_agent):
                self.simulation[i].set_property_value("propulsion/tank/contents-lbs", 1000)
                self.simulation[i].set_property_value("propulsion/tank[1]/contents-lbs", 1000)

                # Set gear up
                self.simulation[i].set_property_value("gear/gear-cmd-norm", 0.0)
                self.simulation[i].set_property_value("gear/gear-pos-norm", 0.0)

                self.simulation[i].run()

        # Get the JSBSim state and save to self.state
        self._get_state()



        ####### todo 这里以下的self.state还没改
        reward = 0
        done = False

        # Check for collision with ground
        if self.state[2] < 10:
            reward = -10
            done = True

        # Check if reached goal
        if np.sqrt(np.sum((self.state[:2] - self.goal[:2])**2)) < self.dg and abs(self.state[2] - self.goal[2]) < self.dg:
            reward = 10
            done = True
        
        return np.hstack([self.state, self.goal]), reward, done, {}
    
    def _get_state(self):
        # Gather all state properties from JSBSim

        #### i 表示对应的观察值是哪一位，需要把所有的飞机观察值加到这一位上
        for i, property in enumerate(STATE_FORMAT):
            # import pdb
            # pdb.set_trace()
            for plane in range(self.num_agent):
                temp_obs = []
                temp_obs.append(self.simulation[plane].get_property_value(property))
            self.state[i] = np.array(temp_obs)
        
        # Rough conversion to meters. This should be fine near zero lat/long
        # self.state[:2] *= RADIUS
    
    def reset(self, seed=None):

        self.current_step = 0

        # Rerun initial conditions in JSBSim
        for i in range(self.num_agent):
            self.simulation[i].run_ic()
            self.simulation[i].set_property_value('propulsion/set-running', -1)
        
        # Generate a new goal
        rng = np.random.default_rng(seed)
        distance = rng.random() * 9000 + 1000
        bearing = rng.random() * 2 * np.pi
        altitude = rng.random() * 3000

        self.goal[:2] = np.cos(bearing), np.sin(bearing)
        self.goal[:2] *= distance
        self.goal[2] = altitude

        # Get state from JSBSim and save to self.state
        self._get_state()

        return np.hstack([self.state, self.goal])

    def render_tacview(self, mode="txt", filepath='./JSBSimRecording.txt.acmi'):

        if mode == "txt":
            if not self._create_records:  ### create_records ---> True or False
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * 0.2  ###记录的时间戳
                f.write(f"#{timestamp:.2f}\n")
                i = 0
                for sim in self.simulation:   ###self._jsbsims.values() 返回每一架飞机的类
                    ### uid lon lat alt roll pitch yaw F16(飞机名称大写) color
                    log_msg = self.log(i)    ### 'A0100,T=119.99999999999999|59.99999999999999|6095.999999998865|1.0652838229764686e-16|3.1805546814635168e-15|360.0,Name=F16,Color=Red'
                    if log_msg is not None:
                        f.write(log_msg + "\n")
                    i += 1
                # for sim in self._tempsims.values():
                #     log_msg = sim.log()
                #     if log_msg is not None:
                #         f.write(log_msg + "\n")
        # TODO: real time rendering [Use FlightGear, etc.]
        else:
            raise NotImplementedError

    def log(self,plane):
        global a
        self.uid = "A0" + str(a+100*plane)
        lon = self.simulation[plane].get_property_value("position/long-gc-rad")
        lat = self.simulation[plane].get_property_value("position/lat-gc-rad")
        alt = self.simulation[plane].get_property_value("position/h-sl-meters")
        roll = self.simulation[plane].get_property_value("attitude/phi-rad")
        pitch = self.simulation[plane].get_property_value("attitude/theta-rad")
        yaw = self.simulation[plane].get_property_value("attitude/psi-rad")
        # lon, lat, alt = self.get_geodetic()
        # roll, pitch, yaw = self.get_rpy() * 180 / np.pi
        log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
        log_msg += f"Name={self.model.upper()},"
        log_msg += f"Color={self.color}"
        return log_msg


    def render(self, mode='human'):
        scale = 1e-3

        if self.viewer is None:
            self.viewer = Viewer(1280, 720)

            f16_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
            self.f16 = RenderObject(f16_mesh)
            self.f16.transform.scale = 1/30
            self.f16.color = 0, 0, .4

            goal_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "cylinder.obj")
            self.cylinder = RenderObject(goal_mesh)
            self.cylinder.transform.scale = scale * 100
            self.cylinder.color = 0, .4, 0

            self.viewer.objects.append(self.f16)
            self.viewer.objects.append(self.cylinder)
            self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit, 21, 1.))
        
        # Rough conversion from lat/long to meters
        x, y, z = self.state[:3] * scale

        self.f16.transform.z = x 
        self.f16.transform.x = -y
        self.f16.transform.y = z

        rot = Quaternion.from_euler(*self.state[9:])
        rot = Quaternion(rot.w, -rot.y, -rot.z, rot.x)
        self.f16.transform.rotation = rot

        # self.viewer.set_view(-y , z + 1, x - 3, Quaternion.from_euler(np.pi/12, 0, 0, mode=1))

        x, y, z = self.goal * scale

        self.cylinder.transform.z = x
        self.cylinder.transform.x = -y
        self.cylinder.transform.y = z

        r = self.f16.transform.position - self.cylinder.transform.position
        rhat = r/np.linalg.norm(r)
        x,y,z = r
        yaw = np.arctan2(-x,-z)
        pitch = np.arctan2(-y, np.sqrt(x**2 + z**2))


        self.viewer.set_view(*(r + self.cylinder.transform.position + rhat + np.array([0, .33, 0])), Quaternion.from_euler(-pitch, yaw, 0, mode=1))


        # print(self.f16.transform.position)

        # rot = Quaternion.from_euler(-self.state[10], -self.state[11], self.state[9], mode=1)
        

        self.viewer.render()

        if mode == 'rgb_array':
            return self.viewer.get_frame()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class PositionReward(gym.Wrapper):
    def __init__(self, env, gain):
        super().__init__(env)
        self.gain = gain
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        displacement = obs[-3:] - obs[:3]
        distance = np.linalg.norm(displacement)
        reward += self.gain * (self.last_distance - distance)
        self.last_distance = distance
        return obs, reward, done, info
    
    def reset(self):
        obs = super().reset()
        displacement = obs[-3:] - obs[:3]
        self.last_distance = np.linalg.norm(displacement)
        return obs

# Create entry point to wrapped environment
def wrap_jsbsim(**kwargs):
    return PositionReward(JSBSimEnv(**kwargs), 1e-2)

# Register the wrapped environment
gym.register(
    id="JSBSim-v0",
    entry_point=wrap_jsbsim,
    max_episode_steps=1200
)

# Short example script to create and run the environment with
# constant action for 1 simulation second.
if __name__ == "__main__":
    from time import sleep
    env = JSBSimEnv()
    env.reset()
    env.render()
    for _ in range(300):
        env.step(np.array([0.05, -0.2, 0, .5]))
        env.render()
        sleep(1/30)
    env.close()