import os
import sys
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from utils import Preprocessor, createVariable, timeSince, merge_loss_dicts, agent_lookup, restore_model


def get_img_state(fq):
    frames = map(lambda x: x.transpose(1, 2, 0), fq)
    border_shape = (84, 10, 1)
    frames = [val for pair in zip(frames, [np.zeros(border_shape)] * len(frames)) for val in pair]
    frames = frames[:-1]
    frames = Image.fromarray(np.hstack(frames)[:, :, -1]).convert('RGB')

    new_size = tuple(np.array(frames.size) * 3)
    frames = frames.resize(new_size, Image.ANTIALIAS)
    return frames


def vis_rand_latents(params):
    agent = agent_lookup(params)

    restore_model(agent, params['restore'], params['use_cuda'])
    if params['use_cuda']:
        agent.cuda()

    agent.eval()

    for i in range(100):
        _, r = agent.sample_latent()
        r = r.numpy()[0]
        img = get_img_state(np.split(r, 4))
        img.save(open('./out/latents/beta2/{0}.png'.format(i), 'wb'))
        # img.save(open('./out/latents/beta1000/{0}.png'.format(i), 'wb'))


def vis_latent_walk(params):
    agent = agent_lookup(params)

    restore_model(agent, params['restore'], params['use_cuda'])
    if params['use_cuda']:
        agent.cuda()

    agent.eval()

    if os.path.exists('./out/latents/beta1000/walk/'):
        shutil.rmtree('./out/latents/beta1000/walk/')
    for i in range(agent.rep_size):
        os.makedirs('./out/latents/beta1000/walk/{0}'.format(i + 1))

    z, r = agent.sample_latent()
    r = r.numpy()[0]
    img = get_img_state(np.split(r, 4))
    img.save(open('./out/latents/beta1000/walk/orig.png', 'wb'))
    deltas = np.linspace(-5, 5, 100)
    for dim in range(agent.rep_size):
        print 'Walking latent space along dimension {0}'.format(dim + 1)
        for i, delta in enumerate(deltas):
            cz = z.clone()
            cz[:, dim] += delta
            img = agent.decode_latent(cz).numpy()[0]
            img = get_img_state(np.split(img, 4))
            img.save(open('./out/latents/beta1000/walk/{0}/{1}.png'.format(dim + 1, i), 'wb'))