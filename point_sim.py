import numpy as np

pos = 0
velocity = 0

class PointSim():
    def __init__(self, action_duration=.2, goal_point=.6, max_velocity=.2, goal_threshold=0.1,  
                 min_pos=-.3, max_pos=.6, max_action=.2):
        self.action_duration = action_duration
        self.goal_point = goal_point
        self.max_velocity = max_velocity
        self.goal_threshold = goal_threshold
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.action_space = np.array([1])
        self.observation_space = np.array([2])
        self.max_action = max_action

    def reset(self):
        global pos
        global velocity
        pos = 0
        velocity = 0
        return self.get_obs()
    
    def get_obs(self):
        global pos
        global velocity
        return [pos, velocity]

    def step(self, action):
        global pos
        global velocity
        
        action = action[0]
        action-=self.max_action
        velocity += action*self.action_duration
        if action > 0:
            velocity = np.clip(velocity, -self.max_velocity, action)
        else:
            velocity = np.clip(velocity, action, self.max_velocity)
        velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
        velocity += np.random.normal(0, 0.005)
        pos += velocity*self.action_duration
        if pos < self.min_pos or pos > self.max_pos:
            pos = np.clip(pos, self.min_pos, self.max_pos)
            velocity = 0
        reward = -np.abs(self.goal_point - pos)
        if np.abs(self.goal_point - pos) < self.goal_threshold:
            done = True
        else:
            done = False
        try:
            pos = pos[0]
            velocity = velocity[0]
        except:
            pass

        return self.get_obs(), reward, done