import gym
from torch import nn
import torch
import numpy as np

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.ln(self.linear(x))

class PolicyNetworkDDPG(nn.Module):
    def __init__(self, stack_frame=4, action_space=None, obs_dim=339, action_dim=22):
        super(PolicyNetworkDDPG, self).__init__()
        self.stack_frame = stack_frame       
        self.action_dim = action_dim 
        self.obs_dim = obs_dim

        # Ref: https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py
        if not action_space:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.tensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.tensor((action_space.high + action_space.low) / 2.)

        self.fc1 = nn.Linear(obs_dim * stack_frame, 1024)
        self.ln = nn.LayerNorm(1024)

        self.hid_mu = nn.Linear(1024, 512)
        self.ln_mu = nn.LayerNorm(512)

        self.mu = nn.Linear(512, action_dim)

        # Initialize the weights and biases
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.hid_mu.weight)
        torch.nn.init.xavier_uniform_(self.mu.weight) 
        torch.nn.init.zeros_(self.hid_mu.bias)
        torch.nn.init.zeros_(self.mu.bias)

    def forward(self, state):
        state = state.reshape(-1, self.stack_frame * self.obs_dim)
        state = nn.functional.elu(self.ln(self.fc1(state)))

        mu = nn.functional.elu(self.ln_mu(self.hid_mu(state)))
        mu = self.mu(mu)
        
        action = torch.tanh(mu) * self.action_scale + self.action_bias

        return action

class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)
        self.counter = 0
        self.model = PolicyNetworkDDPG(action_space=self.action_space, stack_frame=1)
        self.model.load_state_dict(torch.load("110062112_hw4_data", map_location=torch.device('cpu')))
        self.cnt = 0
        self.last_action = None

    def obs_preprocess(self, observation):
        # Convert observation to a vector
        obs = []
        # v_tgt_field
        obs.extend(observation["v_tgt_field"].flatten())
        # pelvis
        for key in observation["pelvis"].keys():
            if key == "vel":
                obs.extend(observation["pelvis"][key])
            else:
                obs.append(observation["pelvis"][key])
        # r_leg
        for key in observation["r_leg"].keys():
            if key == "ground_reaction_forces":
                obs.extend(observation["r_leg"][key])
            else:
                for subkey in observation["r_leg"][key].keys():
                    obs.append(observation["r_leg"][key][subkey])
        # l_leg
        for key in observation["l_leg"].keys():
            if key == "ground_reaction_forces":
                obs.extend(observation["l_leg"][key])
            else:
                for subkey in observation["l_leg"][key].keys():
                    obs.append(observation["l_leg"][key][subkey])

        flattened = np.array(obs).flatten()
        return torch.tensor(flattened, dtype=torch.float32)

    def act(self, observation):
        if self.cnt == 0:
            action = self.model(self.obs_preprocess(observation)).detach().numpy()
            self.last_action = action
        
        action = self.last_action
        self.cnt = 0 if self.cnt == 4 else self.cnt + 1
        return action[0]