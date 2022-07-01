import numpy as np
from tkinter import *
import tkinter as tk
import time, pickle
import threading
from point_sim import PointSim

class App(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        
        self.w = Scale(self.root, from_=0, to=1, orient=HORIZONTAL, tickinterval=.0001, resolution=.0001)
        self.w.pack()


    def callback(self):
        self.root.quit()

    def run(self):
        self.root.update()

    def get_w(self):
        return self.w.get()


app = App()
#app2 = App()
for i in range(100000):
    
    env = PointSim()
    #LunarLanderContinuous-v2
    #agent = Agent(alpha=0.000314854, beta=0.000314854, input_dims=env.observation_space.shape, env=env, batch_size=128,
    #        tau=.02, max_size=50000, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0], reward_scale=1, auto_entropy=True,max_action=.5)
    agent = pickle.load(open("agents\point_minus20_1000.p", "rb" ))
    print(agent.scale)
    #agent.actor.max_action=.1
    n_games = 3000
    rewards = []
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'LunarLanderContinuous.png'

    figure_file = 'plots/' + filename

    best_score = -10000
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
    
    limit_factor = 0.1
    limit_factor2 = 0.0

    render = False
    env_interacts = 0

    for i in range(n_games):
        speed = []
        yvel=[]
        observation = env.reset()
        observation = np.concatenate((observation, [limit_factor]))
        done = False
        score = 0
        episode_interacts=0
        while not done:
            episode_interacts+=1
            app.run()
            #app2.run()
            limit_factor = app.w.get()
            #limit_factor2 = app2.w.get()
            env_interacts+=1
            #if env_interacts%300 == 0:
            #    print("INTERACT", env_interacts)
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            observation_ = np.concatenate((observation_, [limit_factor]))
            score += reward
            speed.append(observation[2])
            yvel.append(observation[0])
            time.sleep(.1)
            #agent.remember(observation, action, reward, observation_, done)
            #if not load_checkpoint:
                #if env_interacts > 1000:
                    #if env_interacts % 128 == 0:
            #    agent.learn(update_params=True)
                    #else:
                        #agent.learn()
            """if(i>=0):
                env.render()
                print(observation[4], observation[6] )
                time.sleep(2)"""
            #if render:
            print(action)
            observation = observation_
            #env.render()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            #if not load_checkpoint:
            #    agent.save_models()

        rewards.append(score)

        #if score > 298:
        #    break
        #print(limit_factor, limit_factor2)
        print(limit_factor)
        #print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score, f"speed: {np.mean(speed)}", f"yvel: {np.mean(yvel)}")
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score, f"speed: {np.mean(speed)}", f"episode_interacts: {episode_interacts}")




mainloop()

