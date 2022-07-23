import torch
from torch._C import device 
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import DiscriminatorNetwork

class Discriminator():
    def __init__(self,lr=.0001, input_dims=(3,),  n_actions=2, max_size=1000000,
        layer1_size=256, layer2_size=256, batch_size=128,  predict_dims=1, disc_input_dims=(1,)):
        self.lr = lr
        self.input_dims = input_dims
        self.max_size = max_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.batch_size = batch_size
        self.predict_dims = predict_dims
        self.disc_input_dims = disc_input_dims

        self.disc_memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.discriminator = DiscriminatorNetwork(lr=lr, input_dims=disc_input_dims,     
            fc1_dims=layer1_size, fc2_dims=layer2_size, prediction_dims=predict_dims)

    def remember(self, state, action, reward, new_state, done):
        self.disc_memory.store_transition(state, action, reward, new_state, done)
    
    def learn(self):
        loss = None
        state, action, reward, new_state, done = \
            self.disc_memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.discriminator.device)
        done = torch.tensor(done).to(self.discriminator.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.discriminator.device)
        state = torch.tensor(state, dtype=torch.float).to(self.discriminator.device)
        action = torch.tensor(action, dtype=torch.float).to(self.discriminator.device)

        limit_factor = torch.clone(state)
        limit_factor = limit_factor[:, -1]
        limit_factor = limit_factor[:,None]

        disc_state = torch.clone(state)
        #disc_state = disc_state[:, :-1]
        disc_state = disc_state[:, 1:2]
        disc_predictions, log_probs, dist = self.discriminator.predict(disc_state, requires_grad=True)
        #print(max(dist.loc), min(dist.loc))
        #print(max(dist.scale), min(dist.scale))
        #print("diff", ((dist.cdf(disc_predictions)-dist.cdf(limit_factor))**2)[0])
        self.discriminator.optimizer.zero_grad()
        loss = (F.mse_loss(disc_predictions, limit_factor)*10+ (torch.nan_to_num((1/(torch.abs((min(dist.loc)-max(dist.loc)))+.0001))))*10) 
        #loss = (F.mse_loss(disc_predictions, limit_factor) + 1/torch.std(dist.loc))*10 
        #print(torch.std(dist.loc))
        #print("mse loss: ", F.mse_loss(disc_predictions, limit_factor))
        loss.backward()
        self.discriminator.optimizer.step()

        if loss is not None:
            return loss, torch.mean(log_probs).item()

    def calculate_reward(self, state, reward, rew_threshold):
        state = torch.tensor((state,), dtype=torch.float).to(self.discriminator.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.discriminator.device)
        disc_state = torch.clone(state)
        limit_factor = torch.clone(state)
        limit_factor = state[:, -1]
        limit_factor = limit_factor[:,None]
        #print(limit_factor)
        # x-vel
        disc_state = disc_state[:, 1:2]
        #print(disc_state)
        disc_predictions, log_probs, dist = self.discriminator.predict(disc_state, requires_grad=False)
        print(disc_predictions)
        log_probs.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        rew=-torch.log(torch.clamp(torch.abs(disc_predictions-limit_factor), min=.0000001, max=.99999))
        if torch.any(torch.isinf(rew)) or torch.any(torch.isnan(rew)):
            print(rew, "rew")    
            print((dist.cdf(disc_predictions)-dist.cdf(limit_factor))**2, "diff")
        #rew = (rew[:,-1]+reward)*10
        
        #print(rew)
        rew = (rew[:,-1])*20
        rew = torch.where(reward<rew_threshold, reward*22, rew)

        return rew.item()

    def save_models(self):
        torch.save(self.discriminator.state_dict(), 'discriminator.pt')

    def load_models(self):
        self.discriminator.load_state_dict(torch.load('discriminator.pt'))