import sys
import gym
import torch
import random

from deep_barley import deep_barley, eval_db_agent, cache_abstraction

from atari_wrappers import make_atari

use_cuda = torch.cuda.is_available()

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED) if not use_cuda else torch.cuda.manual_seed(SEED)


def main():
    agent = 'BARLEY'

    # env_name = 'PongDeterministic-v4'
    env_name = 'BreakoutDeterministic-v4'
    # env_name = 'SeaquestDeterministic-v4'
    print 'Environment: {0}'.format(env_name)

    envs = [make_atari(env_name) for _ in range(1)]
    for i, env in enumerate(envs):
        env.seed(SEED + i)

    state_dim = envs[0].observation_space.shape
    state_dim = state_dim[0] if len(state_dim) == 1 else state_dim
    # print state_dim

    print str(envs[0].unwrapped.get_action_meanings())

    params = {"arch": agent,
              "num_epochs": 100000,
              "learning_rate": 0.01,
              "beta": 100.0,
              "batch_size": 16,
              "state_dim": 4,
              "action_dim": envs[0].action_space.n,
              "use_cuda": use_cuda,
              'print_every': 20,
              'save_every': 10,
              'env_name': env_name,
              'num_episodes': 100,
              'max_steps': 100000,
              'env_render': not use_cuda,
              "use_preproc": True,
              "resize_shape": (84, 84),
              "history": 4,
              "use_luminance": True,
              }

    print sorted(params.iteritems())

    deep_barley(params)
    # eval_db_agent(envs[0], params)
    # cache_abstraction(envs[0], params)

if __name__ == '__main__':
    main()