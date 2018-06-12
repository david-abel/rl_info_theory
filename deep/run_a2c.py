import sys
import gym
import torch
import random

from train import train_agent, train_agent_parallel
from eval import eval_agent_parallel, eval_agent, cache_eval_episode

from atari_wrappers import make_atari, wrap_deepmind

use_cuda = torch.cuda.is_available()

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED) if not use_cuda else torch.cuda.manual_seed(SEED)


def main():
    agent = 'A2C'

    num_envs = 1
    # num_envs = 2
    # num_envs = 4
    # num_envs = 8
    # num_envs = 16

    # env_name = 'PongDeterministic-v4'
    env_name = 'BreakoutDeterministic-v4'
    # env_name = 'SeaquestDeterministic-v4'
    print 'Environment: {0}'.format(env_name)

    envs = [make_atari(env_name) for _ in range(num_envs)]
    # envs = [wrap_deepmind(make_atari(env_name)) for _ in range(num_envs)]
    # envs = [gym.make(env_name) for _ in range(num_envs)]
    for i, env in enumerate(envs):
        env.seed(SEED + i)

    state_dim = envs[0].observation_space.shape
    state_dim = state_dim[0] if len(state_dim) == 1 else state_dim
    # print state_dim

    print str(envs[0].unwrapped.get_action_meanings())

    params = {"arch": agent,
                  "num_episodes": 500000,
                  "max_steps": 100000,
                  "learning_rate": 0.00025,
                  "gamma": 0.99,
                  "beta": 0.01,
                  "lambda": 1.0,
                  "state_dim": 4,
                  "action_dim": envs[0].action_space.n,
                  "print_every": 1,
                  "env_render": not use_cuda,
                  "use_cuda": use_cuda,
                  "use_preproc": True,
                  "resize_shape": (84, 84),
                  "history": 4,
                  "use_luminance": True,
                  'update_freq': 5,
                  # 'update_freq': 50,
                  'action_repeat': 4,
                  'num_envs': num_envs,
                  'save_every': 100,
                  'env_name': env_name,
                  'parallel': True
                  }

    print sorted(params.iteritems())

    # eval_agent(envs[0], params)
    # eval_agent_parallel(envs, params)
    cache_eval_episode(envs[0], params)

if __name__ == '__main__':
    main()
