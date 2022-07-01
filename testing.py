from point_sim import PointSim

env = PointSim()
obs = env.reset()
action = [.1]

done = False
interacts = 0
while not done:
    interacts+=1
    obs, rew, done = env.step(action)
    print(f"obs: {obs}, rew:{rew}, done: {done}")

print(interacts)