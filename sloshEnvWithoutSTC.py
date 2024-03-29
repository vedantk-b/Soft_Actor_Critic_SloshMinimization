
# SLosh enviornment without STC Acods paper code

from argparse import Action
from functools import total_ordering
import math
# from aiohttp import DataQueue
# from cv2 import FlannBasedMatcher
import numpy as np
from stable_baselines3 import PPO
import gym 
from gym import spaces
from numpy import sin, cos, power
import matplotlib.pyplot as plt
import random
from collections import deque

# from scipy.integrate import odeint

class SloshEnv(gym.Env):
    
    def __init__(self,u= 1.2):
        super(SloshEnv, self).__init__()

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low = -1, high = 1, shape=(5,), dtype=np.float64)

        
        self.ms = 1.32 # in kg
        self.M = 10.82 # M=Ml+Mc+Mb+m=10.82
        self.g = 9.8
        self.cd = 0.00030490 # damping coefficient given in equation (1b)
        self.l = 0.052126    #in meter
        self.d = 0       # d is external disturbance in input channel equation 1(a)

        
    def dynamics(self, state, u):

        zeta1 = state[0]
        zeta2 = state[1]
        zeta3 = state[2]
        zeta4 = state[3]
        
        ms = self.ms
        M = self.M
        g = self.g
        cd = self.cd
        l = self.l
        d = self.d
        D = M - ms * power(cos(zeta3),2)


        # __________ EQUATION 5 ____________

        f1_term1 = ms * g * sin(zeta3) * cos(zeta3)
        f1_term2 = (cd/l) * zeta4 * cos(zeta3)
        f1_term3 = ms * l * power(zeta4,2) * sin(zeta3)
        f1 = (f1_term1 + f1_term2 + f1_term3)/D
        b1 = 1/D

        f2_term1 = (g/l) * sin(zeta3)
        f2_term2 = (cd/(ms * power(l,2))) * zeta4
        f2_term3 = (cos(zeta3)/l) * f1
        f2 = -(f2_term1 + f2_term2 + f2_term3)
        b2 = cos(zeta3)/(l * D)

        
        zeta1Dot = zeta2
        zeta2Dot = f1 + b1*(u + d)
        zeta3Dot = zeta4 
        zeta4Dot = f2 + b2*(u + d)

        return [zeta1Dot,zeta2Dot,zeta3Dot,zeta4Dot]


    def step(self, action):
        info = {}    
        action_copy = action
        action = action*3
        self.total_steps += 1
        # T = 0.01
        T = 0.1
        self.u = action
        xDes = 0.175
        xDotDes = 0
        phiDes = 0
        phiDotDes = 0

        ex = self.x - xDes
        exDot = self.xDot - xDotDes
        ephi = self.phi - phiDes
        ephiDot = self.phiDot - phiDotDes

        initCon = [ex,exDot,ephi,ephiDot]  #initial condition in terms of error
        
        dT = 0.0001        
        x = np.array(initCon)
        sol = []

        reward = 0

        for t in np.arange(0,T,dT):
            sol.append(x)
            xDot = self.dynamics(x,self.u)
            xDot = np.array(xDot, dtype=np.float64)
            # print(f"t = {t}")

            x = x + xDot*dT

            # print(f"x of zero is = {x[0]}")
            
            cost = -(10*abs(x[0])+abs(x[2])+abs(x[3]))

            # if (abs(x[0]) < 0.0025 ) and (abs(x[1]) < 1):
            #     cost = cost + 1.5
            reward += cost*0.05

        info["func_rew"] = reward
        
        reward = reward - abs(self.u/3 - self.prev_action)*20

        info["udif_rew"] = reward - info["func_rew"]
        # if abs(self.u - self.prev_action) > 0.05:
        # print(f"act_diff = {abs(self.u/3 - self.prev_action)}")
                
        # if abs(self.u) > 3 :
        #     cost = cost - abs(self.u)*10

        sol = np.array(sol)
        

        # info["sol"] = sol
        return_state = [0,0,0,0]
        
        #return state or state at the end of episode 
        return_state[0] = sol[-1][0]+0.175 #1st element of last entry of sol+0.175
        return_state[1] = xDot[0]          #first element selected which is xDOt
        return_state[2] = sol[-1][2]       #3rd element of last entry of sol
        return_state[3] = sol[-1][3]       #4th element of last entry of sol

        self.x = return_state[0]
        self.xDot = return_state[1]
        self.phi = return_state[2]
        self.phiDot = return_state[3]

        return_state.append(self.prev_action)

        observation = np.array(return_state, dtype=np.float64)

        self.prev_states.append(observation[0]) 

        prev_states_in_range = True

        for i in self.prev_states:
            # if(abs(i - 0.175) > (0.1744/max(1, math.log(self.total_steps, 1.062)))):
            if(abs(i - 0.175) > 0.001):
                reward -= abs(i - 0.175)*10
                prev_states_in_range = False
            else:
                print("100rev hehe")
                reward += 100
        
        info["prev_state_reward"] = reward - info["func_rew"] - info["udif_rew"]
        
        if ((len(self.prev_states) == 10) and (prev_states_in_range == True)) :
            print("50K reward woahahaha!!!")
            reward += 5000
            self.done = True

        if(self.total_steps > 600):
            reward -= 200
            self.done = True
            
        # self.reward = self.total_reward - self.prev_reward
        # self.prev_reward = self.total_reward

        info["terminate_reward"] = reward - info["func_rew"] - info["udif_rew"] - info["prev_state_reward"]

        info["sol"] = sol

        # self.reward = np.float64(self.reward)
        reward = np.float64(reward)
        self.prev_action = action_copy
        return observation, reward, self.done, info
    
    def reset(self):
        self.prev_action = 0
        self.total_steps = 0
        self.prev_states = deque(maxlen=10)
        self.total_reward = 0
        self.prev_reward  = 0
        self.done = False
        self.x = 0.0
        self.xDot = 0.0
        self.phi = 0.0
        self.phiDot = 0.0
        observation = [0.0, 0.0, 0.0, 0.0, 0.0]
        observation = np.array(observation)
        return observation