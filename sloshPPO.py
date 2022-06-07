from argparse import Action
from sys import stderr
# from cv2 import FlannBasedMatcher
import numpy as np
from stable_baselines3 import PPO
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import time

import tensorboard
from sloshEnvSTC import SloshEnv

ti = (time.time())

logdir = "logs/PPO"

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = SloshEnv()
env = Monitor(env)
env.reset()


model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log=logdir)
for i in range(10000):
    model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name=f"PPO{ti}")
    model.save(f"weights/PPO/ppo_{ti}_weight_{(i+751)*10000}_iterations")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)

print(f"mean_reward = {mean_reward} +- {std_reward}")