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

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

agent = pickle.load(open("agents\SA_15rew_speed_10MSEandp1_sparEnvRew_cdfOnly_withMaxDiff_from3500_indexRew_13000.p", "rb" ))

disc_predictions, disc_log_probs, dist = agent.discriminator.predict(torch.tensor([0.4]).to("cuda:0"), requires_grad=False)


print(dist)
n = 100
speeds = np.arange(-.1, 1, step=1/n)
#print(speeds)

x = np.random.randn(n)

x_axis = np.arange(-1, 1, 0.001)

def update(curr):
    if curr == n:
        a.event_source.stop()
    plt.cla()
    bins = np.arange(-4, 4, 0.5)
    # plt.hist(x[:curr], bins = bins)
    try:
        input = np.array([speeds[curr]])
    except:
        input = np.array([speeds[curr-1]])
    input = torch.from_numpy(input).float().cuda()
    #print(input)
    disc_predictions, disc_log_probs, dist = agent.discriminator.predict(input, requires_grad=False)
    mean = dist.loc.item()
    try:
        sd = dist.scale.item()
    except:
        sd = dist.scale.item()
    
    plt.plot(x_axis, norm.pdf(x_axis, mean, sd), label='pdf')
    plt.axis([-1, 1, 0, 10])
    plt.gca().set_title('sampling the normal distribution')
    plt.gca().set_ylabel('frequency')
    plt.gca().set_xlabel('value')
    plt.legend([f"speed: {round(input.item(), 10)}\n mu: {round(dist.loc.item(), 10)}"], loc="upper left")
    plt.gca().annotate('n={}'.format(curr), [3, 27])


fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval = 300)

plt.show()