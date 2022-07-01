from argparse import Action
from sys import stderr
from tabnanny import verbose
# from cv2 import FlannBasedMatcher
import numpy as np
from stable_baselines3 import PPO, SAC
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import time

import tensorboard
from torch import tensor
# from sloshEnvWithoutSTC import SloshEnv
from sloshGymEnvNoSTC import SloshEnv

ti = (time.time())

logdir = "logs/SAC/no_stc/new_env3"

modeldir = "weights/SAC/no_stc/new_env3"

if not os.path.exists(modeldir):
    os.makedirs(modeldir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = SloshEnv()
env = Monitor(env)
env.reset()

model = SAC("MlpPolicy", env=env, verbose=1, tensorboard_log=logdir, device="cuda")
for i in range(10000):
    model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name=f"SACenv3{ti}")
    model.save(f"{modeldir}/sacEnv3_{ti}_weight_{(i)*10000}_iterations")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)

print(f"mean_reward = {mean_reward} +- {std_reward}")