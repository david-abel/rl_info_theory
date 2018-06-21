from __future__ import division

import os
import sys
import time

import shutil
import torch
from torch.utils.data import DataLoader

from networks import VAE
from utils import EpisodeDataset, Preprocessor, createVariable, timeSince, agent_lookup


def loss_gauss(pi_d, pi_phi, mu, logvar, params):
    kld = torch.nn.KLDivLoss(size_average=False)
    recon_loss = kld(torch.log(pi_phi), pi_d) / params['batch_size']
    prior_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    loss = (params['beta'] * recon_loss) + prior_loss
    return loss, recon_loss, prior_loss


def loss_concrete(pi_d, pi_phi, phi, params):
    kld = torch.nn.KLDivLoss(size_average=False)
    recon_loss = kld(torch.log(pi_phi), pi_d) / params['batch_size']
    k = phi.size()[-1]
    prior_loss = kld(torch.log(phi), torch.ones_like(phi) * (1 / k)) / params['batch_size']
    loss = (params['beta'] * recon_loss) + prior_loss
    # print recon_loss, prior_loss
    return loss, recon_loss, prior_loss


def deep_barley(params):
    agent = VAE(params['state_dim'], params['action_dim'])
    agent.train()
    if params['use_cuda']:
        agent = agent.cuda()

    dataset = EpisodeDataset('./out/A2C_{0}_episode.pkl'.format(params['env_name']))
    trainloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(agent.parameters(), lr=params['learning_rate'])
    # optimizer = torch.optim.RMSprop(agent.parameters(), lr=params['learning_rate'])
    for epoch in xrange(1, params['num_epochs'] + 1):
        total_loss = 0.0
        for batch_id, batch in enumerate(trainloader):
            optimizer.zero_grad()
            batch_states, batch_pols = batch['state'], batch['policy']
            if params['use_cuda']:
                batch_pols = batch_pols.cuda()
            if agent.use_concrete:
                pi_phi, _, phi = agent.forward(createVariable(batch_states, use_cuda=params['use_cuda']))
                phi, _ = phi
                loss, r_loss, p_loss = loss_concrete(batch_pols, pi_phi, phi, params)
            else:
                pi_phi, _, rets = agent.forward(createVariable(batch_states, use_cuda=params['use_cuda']))
                mus, logvars = rets
                loss, r_loss, p_loss = loss_gauss(batch_pols, pi_phi, mus, logvars, params)
            loss.backward()
            total_loss += loss.data
            optimizer.step()

            if (batch_id + 1) % params['print_every'] == 0:
                print '\tBatch {} | Total Loss: {:.6f} | R-Loss {:.6f} | P-Loss {:.6f} | \t[{}/{} ({:.0f}%)]' \
                    .format(batch_id + 1, loss.data, r_loss.data, p_loss.data, batch_id * len(batch_states),
                            len(trainloader.dataset), 100. * batch_id / len(trainloader))
        print 'Epoch {} | Total Loss {:.6f}'.format(epoch + 1, total_loss)
        if (epoch + 1) % params['save_every'] == 0 or (epoch + 1) == params['num_epochs']:
            torch.save(agent.state_dict(), './agents/{0}_{1}'.format(params['arch'], params['env_name']))


def eval_db_agent(env, params):
    if params['use_preproc']:
        preprocessor = Preprocessor(params['state_dim'], params['history'], params['use_luminance'],
                                    params['resize_shape'])
        params['state_dim'] = preprocessor.state_shape
    else:
        preprocessor = None

    agent = VAE(params['state_dim'], params['action_dim'])
    if params['use_cuda']:
        agent = agent.cuda()
        agent.load_state_dict(torch.load('./agents/{0}_{1}'.format(params['arch'], params['env_name'])))
    else:
        agent.load_state_dict(
            torch.load('./agents/{0}_{1}'.format(params['arch'], params['env_name']), map_location='cpu'))
    agent.eval()

    agent_steps = 0
    episode_rewards = []
    start = time.time()
    for episode in xrange(1, params['num_episodes'] + 1):
        env_state = env.reset()
        episode_reward = 0.0
        for t in xrange(1, params['max_steps'] + 1):
            if params['env_render']:
                env.render()

            if preprocessor:
                state = preprocessor.process_state(env_state)
            else:
                state = env_state

            var_state = createVariable(state, use_cuda=params['use_cuda'])
            action, state_val = agent.sample_action_eval(var_state)

            reward = 0.0
            for _ in range(1):
                env_state, r, terminal, _ = env.step(action)
                reward += r
                if terminal:
                    break

            episode_reward += reward

            if terminal:
                break

        episode_rewards.append(episode_reward)
        agent_steps += t

        if preprocessor:
            preprocessor.reset()

        print 'Episode {0} | Total Steps {1} | Total Reward {2} | Mean Reward {3} | Total Time {4}' \
            .format(episode, agent_steps, episode_reward, sum(episode_rewards[-100:]) / 100,
                    timeSince(start, episode / params['num_episodes']))


def cache_abstraction(env, params):
    if os.path.exists('./out/{0}'.format(params['env_name'])):
        shutil.rmtree('./out/{0}'.format(params['env_name']))

    if params['use_preproc']:
        preprocessor = Preprocessor(params['state_dim'], params['history'], params['use_luminance'],
                                    params['resize_shape'])
        params['state_dim'] = preprocessor.state_shape
    else:
        preprocessor = None

    agent = VAE(params['state_dim'], params['action_dim'])
    if params['use_cuda']:
        agent = agent.cuda()
        agent.load_state_dict(torch.load('./agents/{0}_{1}'.format(params['arch'], params['env_name'])))
    else:
        agent.load_state_dict(
            torch.load('./agents/{0}_{1}'.format(params['arch'], params['env_name']), map_location='cpu'))
    agent.eval()

    agent_steps = 0
    episode_rewards = []
    start = time.time()
    for episode in xrange(1):
        env_state = env.reset()
        episode_reward = 0.0
        for t in xrange(1, params['max_steps'] + 1):
            if params['env_render']:
                env.render()

            if preprocessor:
                state = preprocessor.process_state(env_state)
            else:
                state = env_state

            var_state = createVariable(state, use_cuda=params['use_cuda'])
            # action, state_val = agent.sample_action_eval(var_state)
            action, state_val, code = agent.sample_action_eval_code(var_state)

            if not os.path.exists('./out/{0}/{1}'.format(params['env_name'], code)):
                os.makedirs('./out/{0}/{1}'.format(params['env_name'], code))
            preprocessor.get_img_state().save('./out/{0}/{1}/{2}.png'.format(params['env_name'], code, t))

            reward = 0.0
            for _ in range(1):
                env_state, r, terminal, _ = env.step(action)
                reward += r
                if terminal:
                    break

            episode_reward += reward

            if terminal:
                break

        episode_rewards.append(episode_reward)
        agent_steps += t

        if preprocessor:
            preprocessor.reset()

        print 'Episode {0} | Total Steps {1} | Total Reward {2} | Mean Reward {3}' \
            .format(episode, agent_steps, episode_reward, sum(episode_rewards[-100:]) / 100)
