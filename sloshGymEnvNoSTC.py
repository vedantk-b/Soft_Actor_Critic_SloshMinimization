
# SLosh enviornment without STC Acods paper code

import numpy as np
from numpy import sin, cos, power
import matplotlib.pyplot as plt
import random
from scipy.integrate import odeint
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

from torch import true_divide

sign = lambda z: abs(z)/z if z!=0 else 0  # One line equivalent of the signum function

u_previous = [1, 1]

class SloshEnv(gym.Env):

    def __init__(self,u= 1.2):
        super(SloshEnv, self).__init__()

        self.action_space = spaces.Box(low=-5, high=5, shape=(1,))
        self.observation_space = spaces.Box(low = -1, high = 1, shape=(4,), dtype=np.float64)

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
        T = 0.01
        initStates = [self.x, self.xDot, self.phi, self.phiDot]
        x = initStates[0]
        xDot = initStates[1]
        phi = initStates[2]
        phiDot = initStates[3]
        
        self.u = action
        u_previous.append(self.u)
        u_previous.remove(u_previous[0])
        xDes = 0.175
        xDotDes = 0
        phiDes = 0
        phiDotDes = 0

        ex = x - xDes
        exDot = xDot - xDotDes
        ephi = phi - phiDes
        ephiDot = phiDot - phiDotDes


        initCon = [ex,exDot,ephi,ephiDot]  #initial condition in terms of error
        
        dT = 0.0001        
        x = np.array(initCon)
        sol = []
        reward = 0

        for t in np.arange(0,T,dT):
            sol.append(x)
            xDot = self.dynamics(x,self.u)
            xDot = np.array(xDot, dtype=np.float32)

            x = x + xDot*dT
            
            # coost = -(1000*abs(x[0])+0.3*abs(x[2])+0.3*abs(x[3]))

            # # print(f"x = {x} \n\n")
            # if (abs(x[0]) < 0.025 ) and (abs(x[1]) < 1):
            #     coost = coost + 1500
            # if abs(self.u) > 3 :
            #     coost = coost - abs(self.u)*1000
            # if abs(self.u - u_previous[-2]) > 0.05:
            #     coost = coost - 100
                
            # reward += coost*0.001
            reward -= (abs(x[0]) + 0.3*abs(x[2]))
                

            
        sol = np.array(sol)
        
        return_state = [0,0,0,0]
        
        #return state or state at the end of episode 
        return_state[0] = sol[-1][0]+0.175 #1st element of last entry of sol+0.175
        return_state[1] = xDot[0]          #first element selected which is xDOt
        return_state[2] = sol[-1][2]       #3rd element of last entry of sol
        return_state[3] = sol[-1][3]       #4th element of last entry of sol

        info["sol"] = sol

        self.x = return_state[0]
        self.xDot = return_state[1]
        self.phi = return_state[2]
        self.phiDot = return_state[3]

        observation = np.array(return_state, dtype=np.float32)

        if(abs(observation[0] - 0.175) < 0.0001):
            self.done = True

        if(observation[0] > 0.175):
            self.done = True
        
        return observation, reward, self.done, info
    
    def reset(self):
        # x0 = round(random.uniform(0.00, 0.250), 4)
        # x1 = round(random.uniform(-0.150, 0.400), 4)
        # x2 = round(random.uniform(-0.25, 0.25), 4)
        # x3 = round(random.uniform(-2.50, 2.50), 4)
        # return [x0,x1,x2,x3]
        self.done = False
        self.x = 0.0
        self.xDot = 0.0
        self.phi = 0.0
        self.phiDot = 0.0
        observation = [0.0, 0.0, 0.0, 0.0]
        observation = np.array(observation)
        return observation