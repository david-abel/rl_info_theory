from __future__ import division
import sys
import cv2
import math
import time
import torch
import numpy as np
import cPickle as pickle

from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset

from networks import A2C, VAEAgent


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def createVariable(x, use_cuda=False):
    if len(x.shape) == 3:
        ret = Variable(torch.from_numpy(x).float().unsqueeze(0))
    else:
        ret = Variable(x.float())
    if use_cuda:
        return ret.cuda()
    else:
        return ret


def merge_loss_dicts(orig, update):
    for k in update:
        orig[k] += update[k]
    return orig


def restore_model(model, restore, use_cuda):
    try:
        if use_cuda:
            model.load_state_dict(torch.load(restore))
        else:
            model.load_state_dict(torch.load(restore, map_location='cpu'))
    except RuntimeError:  # If full reload doesn't work, attempt partial reload
        full_state = model.state_dict()
        if use_cuda:
            full_state.update(torch.load(restore))
        else:
            full_state.update(torch.load(restore, map_location='cpu'))
        model.load_state_dict(full_state)


def agent_lookup(params):
    archs = ['A2C', 'DBAgent', 'DBAgentAE']
    assert params['arch'] in archs

    if params['arch'] == 'A2C':
        print 'Running A2C...'
        return A2C(params['state_dim'], params['action_dim'], parallel=params['parallel'])
    if 'DBAgent' in params['arch']:
        if 'DBAgent' == params['arch']:
            print 'Running episodic DB'
        elif 'DBAgentAE' == params['arch']:
            print 'Running AE for pre-trained DB agent'

        agent = A2C(params['state_dim'], params['action_dim'])
        if params['use_cuda']:
            agent = agent.cuda()
            agent.load_state_dict(torch.load('./agents/A2C_{0}'.format(params['env_name'])))
        else:
            agent.load_state_dict(torch.load('./agents/A2C_{0}'.format(params['env_name']),
                                             map_location='cpu'))
        agent.eval()
        return VAEAgent(agent, params['state_dim'], params['action_dim'])
    else:
        print 'Unknown architecture specified!'
        sys.exit(0)


class Preprocessor:
    def __init__(self, state_shape, history_size, use_luminance=True, resize_shape=None):
        self.history = history_size
        self.use_luminance = use_luminance
        self.resize_shape = resize_shape

        height, width, chans = (84, 84, 4) # state_shape

        if use_luminance:
            chans = 1

        if self.resize_shape:
            height, width = self.resize_shape
        self.queue_state_shape = (chans, height, width)
        self.state_shape = (chans * self.history, height, width)

        self.frame_queue = []
        self.np_transpose = lambda x: x.transpose(2, 0, 1)
        self.resize_img = lambda x: cv2.resize(x, self.resize_shape, interpolation=cv2.INTER_AREA)

        self.reset()

    def compute_luminance(self, img):
        channels = np.dsplit(img, 3)
        lum = np.array(0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2])
        return lum

    def process_state(self, env_state):
        image = env_state
        if self.resize_shape:
            image = self.resize_img(image)

        if self.use_luminance:
            image = self.compute_luminance(image)

        image = self.np_transpose(image)
        self.frame_queue.append(image)
        while len(self.frame_queue) > self.history:
            self.frame_queue.pop(0)

        return np.vstack(self.frame_queue)

    def get_state(self):
        return np.vstack(self.frame_queue)

    def reset(self):
        for _ in range(self.history - 1):
            self.frame_queue.append(np.zeros(self.queue_state_shape))

    def get_img_state(self):
        frames = map(lambda x: x.transpose(1, 2, 0), self.frame_queue)
        border_shape = (84, 10, 1)
        frames = [val for pair in zip(frames, [np.zeros(border_shape)] * len(frames)) for val in pair]
        frames = frames[:-1]
        frames = Image.fromarray(np.hstack(frames)[:, :, -1]).convert('RGB')

        new_size = tuple(np.array(frames.size) * 3)
        frames = frames.resize(new_size, Image.ANTIALIAS)
        return frames


class EpisodeDataset(Dataset):
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        with open(self.pkl_path, 'rb') as pkl_f:
            self.states, self.distros = pickle.load(pkl_f)

        assert self.states.shape[0] == self.distros.shape[0]

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return {"state": self.states[idx], "policy": self.distros[idx]}