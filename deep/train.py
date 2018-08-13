from __future__ import division
import sys
import time
import torch
import random
import numpy as np
from collections import defaultdict

from utils import Preprocessor, createVariable, timeSince, merge_loss_dicts, agent_lookup, restore_model


def train_step_a2c(agent, optimizer, params):
    R = agent.final_state_val
    policy_loss = []
    value_loss = []
    rewards = []
    for r in agent.rewards[::-1]:
        R = r + params['gamma'] * R
        rewards.insert(0, R)

    rewards = torch.Tensor(rewards)

    for (log_prob, value, distro), reward in zip(agent.saved, rewards):
        entropy = params['beta'] * torch.mul(distro, torch.log(distro)).sum()
        policy_loss.append((-log_prob * (reward - value.data[0][0])) + entropy)
        value_loss.append((reward - value) ** 2)

    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum() / len(rewards)
    value_loss = torch.stack(value_loss).sum() / len(rewards)
    loss = policy_loss + 0.5 * value_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm(agent.parameters(), 40)
    optimizer.step()

    del agent.rewards[:]
    del agent.saved[:]
    return policy_loss.data[0], value_loss.data[0]


def train_step(agent, optimizer, params):
    if 'A2C' == params['arch']:
        # print 'Running A2C update'
        return train_step_a2c(agent, optimizer, params)
    else:
        print 'Unknown training scheme specified!'
        sys.exit(0)


def train_step_parallel(agent, optimizer, params):
    if 'A2C' == params['arch']:
        # print 'Running A2C update'
        return train_step_parallel_a2c(agent, optimizer, params)
    elif 'DBAgent' == params['arch']:
        return train_step_parallel_vae(agent, optimizer, params)
    elif 'DBAgentAE' == params['arch']:
        return train_step_parallel_decode(agent, optimizer, params)
    else:
        print 'Unknown training scheme specified!'
        sys.exit(0)


def train_agent(env, params):
    if params['use_preproc']:
        preprocessor = Preprocessor(params['state_dim'], params['history'], params['use_luminance'],
                                    params['resize_shape'])
        params['state_dim'] = preprocessor.state_shape
    else:
        preprocessor = None

    agent = agent_lookup(params)
    agent.train()

    if params['optim'] == 'rms':
        optimizer = torch.optim.RMSprop(agent.parameters(), lr=params['learning_rate'])
    elif params['optim'] == 'adam':
        optimizer = torch.optim.Adam(agent.parameters(), lr=params['learning_rate'])
    else:
        print 'Unknown optimizer specified!'
        sys.exit(0)

    if params['use_cuda']:
        agent = agent.cuda()

    agent_steps = 0
    episode_rewards = []
    start = time.time()
    for episode in xrange(1, params['num_episodes'] + 1):
        env_state = env.reset()
        episode_reward = 0.0
        policy_loss, value_loss = 0.0, 0.0
        num_updates = 0
        for t in xrange(1, params['max_steps'] + 1):
            if params['env_render']:
                env.render()

            if preprocessor:
                state = preprocessor.process_state(env_state)
            else:
                state = env_state

            var_state = createVariable(state, use_cuda=params['use_cuda'])
            action, state_val = agent.sample_action(var_state)

            reward = 0.0
            for _ in range(1):
                env_state, r, terminal, _ = env.step(action)
                reward += r
                if terminal:
                    break

            agent.rewards.append(reward)
            episode_reward += reward

            if terminal:
                agent.final_state_val = 0.0
                break

            if t % params['update_freq'] == 0:
                agent.final_state_val = state_val[0]
                pl, vl = train_step(agent, optimizer, params)
                policy_loss += pl
                value_loss += vl
                num_updates += 1

        episode_rewards.append(episode_reward)
        agent.final_state_val = 0.0
        pl, vl = train_step(agent, optimizer, params)
        policy_loss += pl
        value_loss += vl
        num_updates += 1
        agent_steps += t

        if preprocessor:
            preprocessor.reset()

        if params['arch'] in ['VQ-A2C']:
            visit = len(agent.visited), agent.visited
            agent.visited = set([])
        else:
            visit = 0

        if episode % params['print_every'] == 0:
            print 'Episode {0} | Total Steps {1} | Total Reward {2} | Mean Reward {3} | Policy Loss {4} | Value Loss {6} | Total Time {5} | S_A {7}' \
                .format(episode, agent_steps, episode_reward, sum(episode_rewards[-100:]) / 100,
                        policy_loss / num_updates,
                        timeSince(start, episode / params['num_episodes']), value_loss / num_updates, visit)


def train_step_parallel_a2c(agent, optimizer, params):
    policy_loss = []
    value_loss = []
    total = 0

    for i in range(params['num_envs']):
        if len(agent.rewards[i]) < 2:
            continue

        R = agent.rewards[i][-1]
        rewards = []
        for r in agent.rewards[i][::-1][1:]:
            R = r + params['gamma'] * R
            rewards.insert(0, R)

        A = 0.0
        advantages = []
        for d in agent.deltas[i][::-1]:
            A = d + (params['gamma'] * params['lambda']) * A
            advantages.insert(0, A)

        rewards = torch.Tensor(rewards)
        advantages = torch.Tensor(advantages)

        for (log_prob, value, distro), reward, advantage in zip(agent.saved[i], rewards, advantages):
            entropy = params['beta'] * -torch.mul(distro, torch.log(distro)).sum()
            rv_diff = reward - value
            # policy_loss.append((-log_prob * advantage) - entropy)
            policy_loss.append((-log_prob * (reward - value.data[0][0])) - entropy)
            # policy_loss.append((-log_prob * (reward - value.data[0][0])))
            value_loss.append(torch.mul(rv_diff, rv_diff))
            total += 1

    agent.rewards.clear()
    agent.saved.clear()
    agent.deltas.clear()

    if len(policy_loss) > 0 and len(value_loss) > 0:
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum() / total
        value_loss = torch.stack(value_loss).sum() / total
        loss = policy_loss + 0.5 * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm(agent.parameters(), 40)
        optimizer.step()

        return {'PL': policy_loss.data[0], 'VL': value_loss.data[0]}
    else:
        return {}


def loss_gauss(pi_d, pi_phi, mu, logvar, params):
    kld = torch.nn.KLDivLoss(size_average=False)
    recon_loss = kld(torch.log(pi_phi), pi_d) / params['batch_size']
    prior_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    loss = (params['beta'] * recon_loss) + prior_loss
    return loss, recon_loss, prior_loss


def loss_gauss_indiv(pi_d, pi_phi, mu, logvar):
    kld = torch.nn.KLDivLoss(size_average=False)
    recon_loss = kld(torch.log(pi_phi + 1e-8), pi_d)
    prior_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    return recon_loss, prior_loss


def train_step_parallel_vae(agent, optimizer, params):
    recon_loss = []
    prior_loss = []
    total = 0

    for i in range(params['num_envs']):
        # if len(agent.saved[i]) < params['batch_size']:
        #     continue

        # sample = random.sample(agent.saved[i], params['batch_size'])
        # states = [x[4][0] for x in sample]
        # pi_phis, _, rets, pi_ds, _ = agent.forward(torch.stack(states))
        # mus, logvars = rets
        # for i in range(len(sample)):
        #     pi_phi, mu, logvar, pi_d = pi_phis[i].unsqueeze(0), mus[i].unsqueeze(0),\
        #                                logvars[i].unsqueeze(0), pi_ds[i].unsqueeze(0)
        #     r_loss, p_loss = loss_gauss_indiv(pi_d, pi_phi, mu, logvar)
        #     recon_loss.append(r_loss)
        #     prior_loss.append(p_loss)
        #     total += 1

        for (log_prob, ret, pi_phi, pi_d, _) in agent.saved[i]:
            # entropy = 0.01 * -torch.mul(pi_phi, torch.log(pi_phi)).sum()
            mu, logvar = ret
            r_loss, p_loss = loss_gauss_indiv(pi_d, pi_phi, mu, logvar)
            # print r_loss, p_loss
            recon_loss.append(r_loss)
            # recon_loss.append(r_loss - entropy)
            prior_loss.append(p_loss)
            total += 1

        agent.saved.clear()
        # while len(agent.saved[i]) > 10000:
        #     agent.saved[i].pop(0)

    if len(recon_loss) > 0 and len(prior_loss) > 0:
        optimizer.zero_grad()
        recon_loss = torch.stack(recon_loss)
        prior_loss = torch.stack(prior_loss)
        recon_loss[recon_loss == float('inf')] = 0.0
        prior_loss[prior_loss == float('inf')] = 0.0
        recon_loss = recon_loss.sum() / total
        prior_loss = prior_loss.sum() / total

        loss = (params['beta'] * recon_loss) + prior_loss
        # loss.backward(retain_graph=True)
        loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 40.)
        optimizer.step()
        # print loss.data.item(), recon_loss.data.item(), prior_loss.data.item()
        return {'RL': recon_loss.data.item(), 'PL': prior_loss.data.item()}
    else:
        return {}


def train_step_parallel_decode(agent, optimizer, params):
    recon_loss = []
    mse = torch.nn.MSELoss(size_average=True)

    for i in range(params['num_envs']):
        if len(agent.saved[i]) < params['batch_size']:
            continue

        sample = random.sample(agent.saved[i], params['batch_size'])
        aes = [x[4] for x in sample]
        states = [x[0] for x in aes]
        _, _, _, _, (_, recons) = agent.forward(torch.stack(states))
        recon_loss.append(mse(recons, torch.stack(states)))

        # agent.saved.clear()
        while len(agent.saved[i]) > 10000:
            agent.saved[i].pop(0)

    if len(recon_loss) > 0:
        optimizer.zero_grad()
        recon_loss = torch.stack(recon_loss)
        recon_loss = recon_loss.sum()

        loss = recon_loss
        loss.backward(retain_graph=False)
        optimizer.step()
        print recon_loss.data.item()
        return {'RL': recon_loss.data.item()}
    else:
        return {}


def train_agent_parallel(envs, params):
    preprocessors = []
    for _ in range(params['num_envs']):
        if params['use_preproc']:
            preprocessor = Preprocessor(params['state_dim'], params['history'], params['use_luminance'],
                                        params['resize_shape'])
            params['state_dim'] = preprocessor.state_shape
        else:
            preprocessor = None
        preprocessors.append(preprocessor)

    agent = agent_lookup(params)

    if params['optim'] == 'rms':
        optimizer = torch.optim.RMSprop(agent.parameters(), lr=params['learning_rate'])
    elif params['optim'] == 'adam':
        optimizer = torch.optim.Adam(agent.parameters(), lr=params['learning_rate'])
    else:
        print 'Unknown optimizer specified!'
        sys.exit(0)

    if params['restore'] is not None:
        restore_model(agent, params['restore'], params['use_cuda'])

    if params['use_cuda']:
        agent = agent.cuda()

    agent.train()
    if params['arch'] == 'DBAgentAE':
        agent.eval()

    episode_rewards = []
    start = time.time()
    total_steps = 0
    for episode in xrange(1, params['num_episodes'] + 1):
        env_states = [env.reset() for env in envs]
        states = [preprocessors[i].process_state(env_states[i]) if preprocessors[i] else env_states[i] for i in
                  range(len(envs))]
        env_status = [False for _ in envs]
        episode_reward = [0.0 for _ in envs]
        loss_dict = defaultdict(float)
        num_updates = 0
        for t in xrange(1, params['max_steps'] + 1):

            if reduce(lambda x, y: x and y, env_status):
                break

            for i, env in enumerate(envs):

                if params['env_render']:
                    env.render()

                if env_status[i]:
                    continue

                var_state = createVariable(states[i], use_cuda=params['use_cuda'])
                action, state_val = agent.sample_action(var_state, i=i)

                reward = 0.0
                for _ in range(1):
                    env_states[i], r, terminal, _ = env.step(action)
                    reward += r
                    if terminal:
                        env_status[i] = True
                        break

                episode_reward[i] += reward
                states[i] = preprocessors[i].process_state(env_states[i]) if preprocessors[i] else env_states[i]

            if t % params['update_freq'] == 0:
                l_dict = train_step_parallel(agent, optimizer, params)
                loss_dict = merge_loss_dicts(loss_dict, l_dict)
                num_updates += 1

        for i, env in enumerate(envs):
            agent.rewards[i].append(0.0)

        l_dict = train_step_parallel(agent, optimizer, params)
        loss_dict = merge_loss_dicts(loss_dict, l_dict)
        num_updates += 1

        for p in preprocessors:
            p.reset()

        episode_rewards.extend(episode_reward)

        # Might need this later
        visit = 0

        total_steps += t
        if episode % params['print_every'] == 0:
            print 'Episode {0} | Total Reward {1} | Total Steps {6} | Mean Reward {2} | Losses {3} | Total Time {4} | SA {5} ' \
                .format(episode, episode_reward, sum(episode_rewards[-100:]) / 100,
                        {k: v / num_updates for k, v in loss_dict.iteritems()},
                        timeSince(start, episode / params['num_episodes']), visit, total_steps)

        if episode % params['save_every'] == 0:
            torch.save(agent.state_dict(), './agents/{0}_{1}_{2}_{3}'.format(params['arch'], params['env_name']
                                                                         , int(params['beta']), params['seed']))
