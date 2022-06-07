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
from sloshEnvSTC import SloshEnv

ti = (time.time())

logdir = "logs/SAC/stc"

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = SloshEnv()
env = Monitor(env)
env.reset()


# model = SAC("MlpPolicy", env=env, verbose=1, tensorboard_log=logdir)
model = SAC.load("weights\SAC\stc\sac_1654435735.0582852_weight_10000_iterations", env=env, verbose=1, tensorboard_log=logdir)
for i in range(10000):
    model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name=f"SAC{ti}")
    model.save(f"weights/SAC/ppo_{ti}_weight_{(i)*10000}_iterations")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)

print(f"mean_reward = {mean_reward} +- {std_reward}")