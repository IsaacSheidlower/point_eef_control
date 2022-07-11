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
from discriminator import Discriminator
import copy

discrim = Discriminator(lr=.0001, input_dims=(3,), layer1_size=256, layer2_size=256)
agent2 = pickle.load(open("point_minus20_noCDF_lowLR_20scale_4000.p", "rb" ))
discrim.discriminator = copy.deepcopy(agent2.discriminator)

state = [0, .05, .2]

reward = 0
print(discrim.calculate_reward(state, reward, reward_threshold=.5))