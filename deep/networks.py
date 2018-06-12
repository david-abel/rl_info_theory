import sys
import copy
from collections import defaultdict

import torch
import random
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.distributions import Categorical
from torch.distributions import RelaxedOneHotCategorical


class A2C(nn.Module):
    def __init__(self, state_dim, action_dim, parallel=False):
        super(A2C, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.c1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(7744, 512)

        self.p_fc = nn.Linear(512, self.action_dim)
        self.v_fc = nn.Linear(512, 1)

        if parallel:
            self.saved = defaultdict(list)
            self.rewards = defaultdict(list)
            self.deltas = defaultdict(list)
        else:
            self.saved = []
            self.rewards = []
            self.deltas = []

    def forward(self, state):
        conv = F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(state))))))
        conv_flat = conv.view(1, -1)
        fc_out = F.relu(self.fc(conv_flat))
        action_scores = self.p_fc(fc_out)
        state_value = self.v_fc(fc_out)
        return F.softmax(action_scores, dim=1), state_value

    def forward_embed(self, state):
        conv = F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(state))))))
        conv_flat = conv.view(1, -1)
        fc_out = F.relu(self.fc(conv_flat))
        return fc_out

    def forward_all(self, state):
        conv = F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(state))))))
        conv_flat = conv.view(1, -1)
        fc_out = F.relu(self.fc(conv_flat))
        action_scores = self.p_fc(fc_out)
        state_value = self.v_fc(fc_out)
        return F.softmax(action_scores, dim=1), state_value, fc_out

    def sample_action(self, state, i=None):
        probs, val = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        if i is not None:
            self.saved[i].append((m.log_prob(action), val, probs))
        else:
            self.saved.append((m.log_prob(action), val, probs))
        return action.data[0], val.data[0]

    def sample_action_eval(self, state):
        probs, val = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.data[0], val.data[0]

    def sample_action_distro(self, state):
        probs, val = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.data[0], val.data[0], probs.data[0]

    def get_state_val(self, state):
        _, val = self.forward(state)
        return val


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, parallel=False):
        super(VAE, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rep_size = 50
        self.temperature = 0.75

        self.c1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7744, 512)
        self.fc_m = nn.Linear(512, self.rep_size)
        self.fc_std = nn.Linear(512, self.rep_size)
        self.fc2 = nn.Linear(512, self.rep_size)

        self.p_fc = nn.Linear(self.rep_size, self.action_dim)
        self.v_fc = nn.Linear(self.rep_size, 1)

        self.training = True
        self.use_concrete = True

        if parallel:
            self.saved = defaultdict(list)
            self.rewards = defaultdict(list)
            self.deltas = defaultdict(list)
        else:
            self.saved = []
            self.rewards = []
            self.deltas = []

    def encode(self, state):
        conv = F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(state))))))
        conv_flat = conv.view(state.size()[0], -1)
        fc_out = F.relu(self.fc1(conv_flat))
        return self.fc_m(fc_out), self.fc_std(fc_out)

    def concrete(self, state):
        conv = F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(state))))))
        conv_flat = conv.view(state.size()[0], -1)
        fc_out = self.fc2(F.relu(self.fc1(conv_flat)))
        # print fc_out.data[0].numpy()
        c = torch.clamp(torch.sign(fc_out), 0.0).data[0].cpu().numpy()
        return RelaxedOneHotCategorical(self.temperature, logits=fc_out).sample(), c

    def repr(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, state):
        if self.use_concrete:
            z, c = self.concrete(state)
            # print z, z.max(1)[1]
            if not self.training:
                print ''.join(map(str, map(int, list(c)))), int(z.max(1)[1])
        else:
            mu, logvar = self.encode(state)
            z = self.repr(mu, logvar)
        action_scores = self.p_fc(z)
        state_value = self.v_fc(z)
        ret = (z, c) if self.use_concrete else (mu, logvar)
        return F.softmax(action_scores, dim=1), state_value, ret

    def sample_action(self, state, i=None):
        probs, val = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        if i is not None:
            self.saved[i].append((m.log_prob(action), val, probs))
        else:
            self.saved.append((m.log_prob(action), val, probs))
        return action.data[0], val.data[0]

    def sample_action_eval(self, state):
        probs, val, _ = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        # action = probs.max(1)[1]
        return action.data[0], val.data[0]

    def sample_action_eval_code(self, state):
        probs, val, r = self.forward(state)
        z, c = r
        m = Categorical(probs)
        action = m.sample()
        # action = probs.max(1)[1]
        # return action.data[0], val.data[0], ''.join(map(str, map(int, list(c))))
        return action.data[0], val.data[0], int(z.max(1)[1])

    def flip_training(self):
        self.training = not self.training
