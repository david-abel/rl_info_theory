from __future__ import division
import sys
import time
import torch
import numpy as np
import cPickle as pickle

from utils import Preprocessor, createVariable, timeSince, agent_lookup, restore_model


def eval_agent(env, params):
    if params['use_preproc']:
        preprocessor = Preprocessor(params['state_dim'], params['history'], params['use_luminance'],
                                    params['resize_shape'])
        params['state_dim'] = preprocessor.state_shape
    else:
        preprocessor = None

    agent = agent_lookup(params)

    if params['use_cuda']:
        agent = agent.cuda()
        agent.load_state_dict(torch.load('./agents/{0}_{1}'.format(params['arch'], params['env_name'])))
    else:
        agent.load_state_dict(torch.load('./agents/{0}_{1}'.format(params['arch'], params['env_name']), map_location='cpu'))

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

        if episode % params['print_every'] == 0:
            print 'Episode {0} | Total Steps {1} | Total Reward {2} | Mean Reward {3} | Total Time {4}' \
                .format(episode, agent_steps, episode_reward, sum(episode_rewards[-100:]) / 100,
                        timeSince(start, episode / params['num_episodes']))


def eval_agent_parallel(envs, params):
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

    restore_model(agent, params['restore'], params['use_cuda'])
    if params['use_cuda']:
        agent.cuda()

    agent.eval()

    episode_rewards = []
    start = time.time()
    for episode in xrange(1, params['num_episodes'] + 1):
        env_states = [env.reset() for env in envs]
        states = [preprocessors[i].process_state(env_states[i]) if preprocessors[i] else env_states[i] for i in
                  range(len(envs))]
        env_status = [False for _ in envs]
        episode_reward = [0.0 for _ in envs]
        for t in xrange(1, params['max_steps'] + 1):

            if reduce(lambda x, y: x and y, env_status):
                break

            for i, env in enumerate(envs):

                if params['env_render']:
                    env.render()

                if env_status[i]:
                    continue

                var_state = createVariable(states[i], use_cuda=params['use_cuda'])
                action, state_val = agent.sample_action_eval(var_state)

                reward = 0.0
                for _ in range(1):
                    env_states[i], r, terminal, _ = env.step(action)
                    reward += r
                    if terminal:
                        env_status[i] = True
                        break
                #
                episode_reward[i] += reward
                states[i] = preprocessors[i].process_state(env_states[i]) if preprocessors[i] else env_states[i]

        for p in preprocessors:
            p.reset()

        episode_rewards.extend(episode_reward)

        if episode % params['print_every'] == 0:
            print 'Episode {0} | Total Reward {1} | Mean Reward {2} | Total Time {3} ' \
                .format(episode, episode_reward, sum(episode_rewards[-100:]) / 100,
                        timeSince(start, episode / params['num_episodes']))


def cache_eval_episode(env, params):
    cache_states, cache_distros = [], []

    if params['use_preproc']:
        preprocessor = Preprocessor(params['state_dim'], params['history'], params['use_luminance'],
                                    params['resize_shape'])
        params['state_dim'] = preprocessor.state_shape
    else:
        preprocessor = None

    agent = agent_lookup(params)

    if params['use_cuda']:
        agent = agent.cuda()
        agent.load_state_dict(torch.load('./agents/{0}_{1}'.format(params['arch'], params['env_name'])))
    else:
        agent.load_state_dict(torch.load('./agents/{0}_{1}'.format(params['arch'], params['env_name']), map_location='cpu'))

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
            action, state_val, distro = agent.sample_action_distro(var_state)
            cache_states.append(state)
            cache_distros.append(distro.cpu().numpy())

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

        if episode % params['print_every'] == 0:
            print 'Episode {0} | Total Steps {1} | Total Reward {2} | Mean Reward {3}' \
                .format(episode, agent_steps, episode_reward, sum(episode_rewards[-100:]) / 100)

    cache_states, cache_distros = np.array(cache_states), np.array(cache_distros)
    pickle.dump((cache_states, cache_distros), open('./out/{0}_{1}_episode.pkl'.format(params['arch'], params['env_name']), 'wb'), -1)