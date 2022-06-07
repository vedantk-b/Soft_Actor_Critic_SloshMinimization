
# SLosh enviornment without STC Acods paper code

from argparse import Action
# from cv2 import FlannBasedMatcher
import numpy as np
from stable_baselines3 import PPO
import gym 
from gym import spaces
from numpy import sin, cos, power
import matplotlib.pyplot as plt
import random

from scipy.integrate import odeint

u_previous = [1, 1]

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

        
    def dynamics(self, state, t, u):

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


    def simulate(self, initStates, T, action, done):
        x = initStates[0]
        xDot = initStates[1]
        phi = initStates[2]
        phiDot = initStates[3]
        
        
        #print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&',u_previous)
        #print(action)
        #print(initStates)
        self.u = action
        u_previous.append(self.u)
        u_previous.remove(u_previous[0])
        #print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&',u_previous)
        #print(self.u)
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
        #print(x)
        sol = []
        reward = 0
        #u=[]   # control input
  
        for t in np.arange(0,T,dT):
            sol.append(x)
            #xDot = self.dynamics(x,t)
            #print(self.u)
            xDot = self.dynamics(x,t,self.u)
            xDot = np.array(xDot)

            x = x + xDot*dT
            #coost = round(-(.40*abs(x[0])+0.3*abs(x[2])+0.3*abs(x[3])),3); # intial cost function
            
            coost = round(-(1000*abs(x[0])+0.3*abs(x[2])+0.3*abs(x[3])),3);
            #coost = round(-(100*abs(x[0])),3);
            #if x[0]>0:
                #coost = coost-10000
            # adding the slope idea of the cart position x[0]= x-xd # duration is tps=0.01
            #current(x-xd) -initial (x-xd)
            # if x[0] > 0 & (x[0] > sol[-1][0]):     
            #     coost = coost-20
                
            # if x[0] < 0 & (x[0] < sol[-1][0]):
            #     coost = coost-20
                
            # if x[0] > 0 & (x[0] < sol[-1][0]):
            #     coost = coost+10
                
            # if x[0] < 0 & (x[0] > sol[-1][0]):
            #     coost = coost+10
 
            if (abs(x[0]) < 0.025 ) and (abs(x[1]) < 1):
                coost = coost + 1500
            # if abs(self.u) > 4:
            #     coost = coost -100
            if abs(self.u) > 3 :
                coost = coost - abs(self.u)*1000
            #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', u_previous[-2], self.u)
            if abs(self.u - u_previous[-2]) > 0.05:
                coost = coost - 100
                
            reward += coost*0.001
                

            
        t = np.arange(0,T,dT)
        sol = np.array(sol)
        
        # if abs(reward)< 100:
        #     done = True
        #     reward += 100
        # else:
        #     done = False
            
        
        return_state = [0,0,0,0]
        
        #return state or state at the end of episode 
        return_state[0] = sol[-1][0]+0.175 #1st element of last entry of sol+0.175
        return_state[1] = xDot[0]          #first element selected which is xDOt
        return_state[2] = sol[-1][2]       #3rd element of last entry of sol
        return_state[3] = sol[-1][3]       #4th element of last entry of sol
       

        return return_state,reward,done,t,sol
    
    def reset(self):
        x0 = round(random.uniform(0.00, 0.250), 4)
        x1 = round(random.uniform(-0.150, 0.400), 4)
        x2 = round(random.uniform(-0.25, 0.25), 4)
        x3 = round(random.uniform(-2.50, 02.50), 4)
        return [x0,x1,x2,x3]
        

    