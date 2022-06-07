
# SLosh enviornment without STC Acods paper code

from argparse import Action
from aiohttp import DataQueue
# from cv2 import FlannBasedMatcher
import numpy as np
from stable_baselines3 import PPO
import gym 
from gym import spaces
from numpy import sin, cos, power
import matplotlib.pyplot as plt
import random
from collections import deque

from scipy.integrate import odeint

u_previous = [1, 1]
# u_previous = [0, 0]

class SloshEnv(gym.Env):
    
    def __init__(self,u= 1.2):
        super(SloshEnv, self).__init__()

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low = -100, high = 100, shape=(4,), dtype=np.float64)

        
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
        action *= 10
        self.total_steps += 1
        T = 0.01
        self.u = action
        u_previous.append(self.u)
        u_previous.remove(u_previous[0])
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

        for t in np.arange(0,T,dT):
            sol.append(x)
            xDot = self.dynamics(x,self.u)
            xDot = np.array(xDot, dtype=np.float64)
            # print(f"t = {t}")

            x = x + xDot*dT

            # print(f"x of zero is = {x[0]}")
            
            cost = -(10*abs(x[0])+0.3*abs(x[2])+0.3*abs(x[3]))

            if (abs(x[0]) < 0.0025 ) and (abs(x[1]) < 1):
                cost = cost + 15

            if abs(self.u) > 3 :
                cost = cost - abs(self.u)*10

            if abs(self.u - u_previous[-2]) > 0.05:
                cost = cost - 10
                
            # print(f"cost = {cost}")
                
            self.total_reward += cost
                

        sol = np.array(sol)
        
        info = {}    

        info["sol"] = sol
        
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

        observation = np.array(return_state)

        self.prev_states.append(observation[0])

        prev_states_in_range = True

        for i in self.prev_states:
            if(abs(i - 0.175) > 0.01):
                self.total_reward -= 200
                prev_states_in_range = False
        
            
        if ((len(self.prev_states) == 20) and (prev_states_in_range == True)) :
            print(f"20K reward hehe")
            self.total_reward += 20000
            self.done = True

        if(self.total_steps > 1000):
            self.total_reward -= 5000
            self.done = True
            
        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward
        
        self.reward = np.float64(self.reward)
        return observation, self.reward, self.done, info
    
    def reset(self):
        self.total_steps = 0
        self.prev_states = deque(maxlen=20)
        self.total_reward = 0
        self.prev_reward  = 0
        self.done = False
        self.x = 0.0
        self.xDot = 0.0
        self.phi = 0.0
        self.phiDot = 0.0
        observation = [0.0, 0.0, 0.0, 0.0]
        observation = np.array(observation)
        return observation