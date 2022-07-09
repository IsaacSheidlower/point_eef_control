from point_sim import PointSim
import numpy as np

env = PointSim()
obs = env.reset()
obs = np.concatenate((obs, [.8]))
#action = [.521]

agent = pickle.load(open("agents\point_minus20_noCDF_20scale_1600.p", "rb" ))

done = False
interacts = 0
while not done:
    interacts+=1
    action = agent.choose_action(obs)
    obs, rew, done = env.step(action)
    if done:
        break
    #print(f"obs: {obs}, rew:{rew}, done: {done}")

print(interacts)