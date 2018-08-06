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
        conv_flat = conv.view(state.size(0), -1)
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
        self.rep_size = 25
        self.hidden_size = 100
        self.temperature = 0.25

        self.c1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7744, 512)
        self.fc_m = nn.Linear(512, self.rep_size)
        self.fc_std = nn.Linear(512, self.rep_size)
        self.fc2 = nn.Linear(512, self.rep_size)

        self.fc3 = nn.Linear(self.rep_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.p_fc = nn.Linear(self.hidden_size, self.action_dim)
        self.v_fc = nn.Linear(self.hidden_size, 1)

        # self.p_fc = nn.Linear(self.rep_size, self.action_dim)
        # self.v_fc = nn.Linear(self.rep_size, 1)

        self.training = True
        # self.use_concrete = True
        self.use_concrete = False

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
            # if not self.training:
            # print ''.join(map(str, map(int, list(c)))), int(z.max(1)[1])
        else:
            mu, logvar = self.encode(state)
            z = self.repr(mu, logvar)
        # action_scores, state_value = self.p_fc(z), self.v_fc(z)
        action_scores, state_value = self.p_fc(F.relu(self.fc4(F.relu(self.fc3(z))))), torch.tensor([0.0])
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
        # print probs
        t = 75.
        # s = torch.exp(probs * t) / torch.sum(torch.exp(probs * t))
        # print s
        # probs = s
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
        return action.data[0], val.data[0], ''.join(map(str, map(int, list(c))))
        # return action.data[0], val.data[0], int(z.max(1)[1])

    def flip_training(self):
        self.training = not self.training


class VAEAgent(nn.Module):
    def __init__(self, base_agent, state_dim, action_dim, parallel=True):
        super(VAEAgent, self).__init__()
        self.base_agent = base_agent
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rep_size = 25
        self.hidden_size = 100
        self.temperature = 0.25

        self.c1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7744, 512)
        self.fc_m = nn.Linear(512, self.rep_size)
        self.fc_std = nn.Linear(512, self.rep_size)
        self.fc2 = nn.Linear(512, self.rep_size)

        self.fc3 = nn.Linear(self.rep_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.p_fc = nn.Linear(self.hidden_size, self.action_dim)

        self.r_fc1 = nn.Linear(self.rep_size, 512)
        self.r_fc2 = nn.Linear(512, 7744)
        self.dc1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dc2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2)
        self.dc3 = nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4)

        # self.dc1 = nn.ConvTranspose2d(self.rep_size, 64, kernel_size=5, stride=1)
        # self.dc2 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1)
        # self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        # self.dc4 = nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4)

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

    def decode(self, z):
        dfc = F.relu(self.r_fc2(F.relu(self.r_fc1(z)))).view(z.size(0), 64, 11, 11)
        deconv = self.dc3(F.relu(self.dc2(F.relu(self.dc1(dfc)))))
        return deconv

    def repr(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, state):
        base_p, base_v = self.base_agent.forward(state)
        base_p, base_v = base_p.detach(), base_v.detach()
        mu, logvar = self.encode(state)
        z = self.repr(mu, logvar)

        recon = self.decode(z.detach())
        action_scores, state_value = self.p_fc(F.relu(self.fc4(F.relu(self.fc3(z))))), torch.tensor([0.0])
        ret = (mu, logvar)
        ret_ae = (state, recon)
        return F.softmax(action_scores, dim=1), state_value, ret, base_p, ret_ae

    def sample_action(self, state, i=None):
        probs, val, ret, demo, ret_ae = self.forward(state)
        state, _ = ret_ae
        # print probs
        m = Categorical(demo)
        action = m.sample()
        if i is not None:
            self.saved[i].append((m.log_prob(action), ret, probs, demo, state))
        else:
            self.saved.append((m.log_prob(action), ret, probs, demo, state))
        return action.data[0], val.data[0]

    def sample_action_eval(self, state):
        probs, val, ret, demo, _ = self.forward(state)
        # print probs
        # soft = True
        soft = False
        if soft:
            t = 75.
            s = torch.exp(probs * t) / torch.sum(torch.exp(probs * t))
            # print s
            probs = s
        m = Categorical(probs)
        action = m.sample()
        # action = probs.max(1)[1]
        return action.data[0], val.data[0]

    def get_state_val(self, state):
        _, val, _, _, _ = self.forward(state)
        return val

    def sample_latent(self):
        z = torch.randn_like(torch.zeros((1, self.rep_size)))
        recon = self.decode(z)
        return z.detach(), recon.detach()

    def decode_latent(self, z):
        return self.decode(z).detach()