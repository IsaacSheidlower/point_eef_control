import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 4,3
from matplotlib.animation import FuncAnimation

from dis import dis
import pandas as pd
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics


import numpy as np
import time, pickle
import torch
from point_sim import PointSim

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

agent = pickle.load(open("agents\point_minus20_noCDF_lowLR_smallDiscBuffer_800.p", "rb" ))

lf = .2
lf2 = .2

env = PointSim()
observation = env.reset()
observation = np.concatenate((observation, [lf]))

env2 = PointSim()
observation2 = env2.reset()
observation2 = np.concatenate((observation2, [lf2]))

env1_count = 0 
env2_count = 0

env_interacts = 0
for i in range(100):
   
    action = agent.choose_action(observation)
    #action2 = agent.choose_action(observation2)
    #except:
    #    pass
    observation_, reward, done = env.step(action)
    #observation2_, reward2, done2 = env2.step(action2)
    #print(f"action: {action}, action2: {action2}")
    #print(f"observation: {observation[1]}, observation2: {observation2[1]}")
    time.sleep(.01)
    env_interacts+=1
    if done:
        observation = env.reset()
        env1_count += 1
        print(env_interacts)
        env_interacts = 0
    #if done2:
    #    observation2 = env2.reset()
    #    env2_count += 1
    y = .7
    y2 = .3
    plt.axis([0, .6, 0, 1])
    plt.scatter(observation[0], y)
    #plt.scatter(observation2[0], y2)
    plt.pause(0.05)
    plt.clf()
    observation = observation_
    observation = np.concatenate((observation, [lf]))
    #observation2 = observation2_
    #observation2 = np.concatenate((observation2, [lf2]))

plt.show()

print(env1_count, env2_count)
