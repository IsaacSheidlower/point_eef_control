import numpy as np

pos = .2
velocity = 0
max_acceleration = 0.1 # max change in velocity per time step

class PointSim():
    def __init__(self, action_duration=.2, goal_point=.6, max_velocity=.2, goal_threshold=0.1,  
                 min_pos=-.3, max_pos=.6, max_action=.3):
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
        pos = .2
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
        action -= .5
        action = action*2
        action = action*self.max_action
        #print("ACTION: ", action, "VELOCITY: ", velocity)

        goal_velocity = action
        curr_velocity = velocity
        if (curr_velocity <= goal_velocity):
            curr_velocity += max_acceleration * self.action_duration
        elif (curr_velocity > goal_velocity):
            curr_velocity -= max_acceleration * self.action_duration
        
        # velocity += action*self.action_duration
        # if action > 0:
        #     velocity = np.clip(velocity, -self.max_velocity, action)
        # else:
            # velocity = np.clip(velocity, action, self.max_velocity)
        velocity = np.clip(curr_velocity, -self.max_velocity, self.max_velocity)
        velocity += np.random.normal(0, 0.01)

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