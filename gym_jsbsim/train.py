import gym
import jsbsim_gym.jsbsim_gym # This line makes sure the environment is registered
from os import path
# from jsbsim_gym.features import JSBSimFeatureExtractor
# from stable_baselines3 import SAC
import torch as th

# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

#
# class JSBSimFeatureExtractor(BaseFeaturesExtractor):
#
#     def __init__(self, observation_space):
#         super().__init__(observation_space, 17)
#
#     def forward(self, observations):
#         # Unpack
#         position = observations[:, :3]
#         mach = observations[:, 3:4]
#         alpha_beta = observations[:, 4:6]
#         angular_rates = observations[:, 6:9]
#         phi_theta = observations[:, 9:11]
#         psi = observations[:, 11:12]
#         goal = observations[:, 12:]
#
#         # Transform position
#         displacement = goal - position
#         distance = th.sqrt(th.sum(displacement[:, :2] ** 2, 1, True))
#         dz = displacement[:, 2:3]
#         altitude = position[:, 2:3]
#         abs_bearing = th.atan2(displacement[:, 1:2], displacement[:, 0:1])
#         rel_bearing = abs_bearing - psi
#
#         # We normalize distance this way to bound it between 0 and 1
#         dist_norm = 1 / (1 + distance * 1e-3)
#         # Normalize these by approximate flight ceiling
#         dz_norm = dz / 15000
#         alt_norm = altitude / 15000
#
#         # Angles to Sine/Cosine pairs
#         cab, sab = th.cos(alpha_beta), th.sin(alpha_beta)
#         cpt, spt = th.cos(phi_theta), th.sin(phi_theta)
#         cr, sr = th.cos(rel_bearing), th.sin(rel_bearing)
#
#         return th.concat([dist_norm, dz_norm, alt_norm, mach, angular_rates, cab, sab, cpt, spt, cr, sr], 1)
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
# policy_kwargs = dict(
#     features_extractor_class=JSBSimFeatureExtractor
# )

# env = gym.make("JSBSim-v0")
#
# log_path = path.join(path.abspath(path.dirname(__file__)), 'logs')
#
# try:
#     model = SAC('MlpPolicy', env, verbose=1,
#                 policy_kwargs=policy_kwargs,
#                 tensorboard_log=log_path,
#                 gradient_steps=-1,
#                 device='cuda')
#     model.learn(3000000)
# finally:
#     model.save("models/jsbsim_sac")
#     model.save_replay_buffer("models/jsbsim_sac_buffer")