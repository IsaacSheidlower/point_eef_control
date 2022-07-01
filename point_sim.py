from operator import pos
from re import M
import numpy as np

pos = 0
velocity = 0

class PointSim():
    def __init__(self, action_duration=.2, goal_point=5, max_velocity=2, goal_threshold=0.1):
        self.action_duration = action_duration
        self.goal_point = goal_point
        self.max_velocity = max_velocity
        self.goal_threshold = goal_threshold

    def reset(self):
        global pos
        global velocity
        pos = 0
        velocity = 0
        return pos
    
    def get_obs(self):
        global pos
        global velocity
        return np.array([pos, velocity])

    def step(self, action):
        global pos
        global velocity
        velocity += action
        velocity = np.clip(pos, -self.max_velocity, self.max_velocity)
        velocity += np.random.normal(0, 0.05)
        pos += velocity
        reward = -np.abs(self.goal_point - pos)

        if np.abs(self.goal_point - pos) < self.goal_threshold:
            done = True
        else:
            done = False

        return self.get_obs(), reward, done