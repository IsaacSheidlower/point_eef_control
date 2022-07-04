"""
This code mainly follows a Soft-Actor Critic YouTube tutorial found at:
https://www.youtube.com/watch?v=ioidsRlf79o&t=2649s
Channel name: Machine Learning with Phil

Any modifiations are made by the AABL Lab.
"""

from ast import expr_context
from point_sim import PointSim
import numpy as np
from sac_torch_costom_new import Agent
import time
import pickle

from tensorboardX import SummaryWriter

if __name__ == '__main__':
    env = PointSim()
    #BipedalWalker-v3
    print(env.observation_space.shape)
    in_dims = (3,)
    #agent = Agent(alpha=.003, beta=.003, disc_lr=.001, input_dims=in_dims, env=env, batch_size=256, disc_layer1_size=256, disc_layer2_size=256,
    #        tau=.02, max_size=100000, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0], reward_scale=15, auto_entropy=True, disc_input_dims=(1,), predict_dims=1)

    agent = Agent(alpha=.003, beta=.003, disc_lr=.001, input_dims=in_dims, env=env, batch_size=256, disc_layer1_size=256, disc_layer2_size=256,
        tau=.02, max_size=100000, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0], reward_scale=15, auto_entropy=True, disc_input_dims=(1,), predict_dims=1)
    #agent = pickle.load(open("/content/drive/MyDrive/costum_rl_agents/SA_15rew_speed_10MSEandp1_sparEnvRew_cdfOnly_withMaxDiff_from3500_indexRew_8000.p", "rb" ))
    #agent.actor.max_action=1
    n_games = 2001
    rewards = []
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    #filename = 'LunarLanderContinuous.png'

    #figure_file = 'plots/' + filename

    best_score = -10000
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    render = False
    env_interacts = 0

    writer = SummaryWriter()
    #action = env.action_space.sample()

    old_reward = 0

    for i in range(0, n_games):
        observation = env.reset()
        done = False
        score = 0
        limit_factor = np.random.uniform(low=0, high=1)
        #limit_factor2 = np.random.uniform(low=0, high=1)
        observation = np.concatenate((observation, [limit_factor]))
        episode_interacts=0
        while not done:
            env_interacts+=1
            episode_interacts+=1
            #agent.actor.max_action = limit_factor
            #try:
            action = agent.choose_action(observation)
            #except:
            #    pass
            observation_, reward, done = env.step(action)
            observation_ = np.concatenate((observation_, [limit_factor]))
            score += reward
            temp_rew = reward
            if reward <= old_reward:
                reward = -20
            else:
                reward = 0
            agent.remember(observation, action, reward, observation_, done)
            old_reward = temp_rew
            #if not load_checkpoint:
                #if env_interacts > 1000:
            #if env_interacts % 400:
            #    limit_factor = np.random.uniform(low=0, high=1)
            observation[-1] = limit_factor
            if env_interacts % 100 == 0:
                try:
                    act_loss, disc1_loss, disc1_log_probs, \
                        disc1_crit, entropy = agent.learn(update_params=True, update_disc=True)
                    if act_loss is not None and disc1_loss is not None:
                        writer.add_scalar("Loss/act_new", np.mean(act_loss.item()),env_interacts)
                        writer.add_scalar("Loss/disc1_loss_new", np.mean(disc1_loss.item()),env_interacts)
                        writer.add_scalar("Loss/disc1_crit_new", disc1_crit,env_interacts)
                        writer.add_scalar("Loss/disc1_log_pob_new", disc1_log_probs,env_interacts)
                        writer.add_scalar("Loss/entropy", entropy,env_interacts)
                        
                except:
                    pass
            else:        
                try:
                    act_loss, _ = agent.learn()
                    act_loss, _ = agent.learn()
                    #if act_loss is not None:
                        #writer.add_scalar(f"Loss/act_loss_366_14", np.mean(act_loss.item()),env_interacts)
                        #writer.add_scalar("Loss/disc_loss", np.mean(disc_loss.item()),env_interacts)
                except Exception as e:
                    print(e)
                    raise

            #if env_interacts % 100 == 0:
                #limit_factor = np.random.uniform(low=0, high=1)
                #limit_factor2 = np.random.uniform(low=0, high=1)

            #if render:
            #env.render()
            observation = observation_
            if episode_interacts > 200:
                done = True
            #limit_factor = np.random.uniform(low=.4, high=1)
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

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f, ' % avg_score, "limit_factor %.2f" % limit_factor, ", interacts, ", episode_interacts)

        if i % 100 == 0 and i > 500:
            pickle.dump(agent, open( f"agents/point_minus20_doubleLearn_20scale_{i}.p", "wb" ) )
    
    #pickle.dump(agent, open( "speedAngle13rew_visi5_3500.p", "wb" ) )
    #np.save("BP_sac_2000", rewards)
