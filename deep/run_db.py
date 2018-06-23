import sys
import gym
import torch
import random
import argparse

from eval import eval_agent_parallel
from train import train_agent_parallel
from deep_barley import deep_barley, eval_db_agent, cache_abstraction

from atari_wrappers import make_atari

use_cuda = torch.cuda.is_available()

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED) if not use_cuda else torch.cuda.manual_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str, help="Choose between train, trainr, eval")
    parser.add_argument("-agent", type=str, help="Choose DBAgent or DBAgentAE")
    parser.add_argument("--restore", type=str, help="Provide path to saved model for trainr or eval modes")
    args = parser.parse_args()
    print 'CLI args: {0}'.format(args)

    return args.agent, args.mode, args.restore


def main():
    agent, mode, restore = parse_args()

    assert mode is not None and agent is not None

    # agent = 'BARLEY'
    # agent = 'DBAgent'
    # agent = 'DBAgentAE'

    # env_name = 'PongDeterministic-v4'
    env_name = 'BreakoutDeterministic-v4'
    # env_name = 'SeaquestDeterministic-v4'
    print 'Environment: {0}'.format(env_name)

    envs = [make_atari(env_name) for _ in range(1)]
    for i, env in enumerate(envs):
        env.seed(SEED + i)

    print str(envs[0].unwrapped.get_action_meanings())

    params = {"arch": agent,
              "learning_rate": 0.001,
              "beta": 100.0,
              "batch_size": 16,
              "state_dim": 4,
              "action_dim": envs[0].action_space.n,
              "use_cuda": use_cuda,
              'print_every': 1,
              'save_every': 10,
              'env_name': env_name,
              'num_episodes': 1500,
              'max_steps': 100000,
              'env_render': not use_cuda,
              "use_preproc": True,
              "resize_shape": (84, 84),
              "history": 4,
              "use_luminance": True,
              "optim": 'adam',
              "update_freq": 16,
              "num_envs": 1,
              "gamma": 0.99,
              "restore": None
              }

    print sorted(params.iteritems())

    # deep_barley(params)
    # eval_db_agent(envs[0], params)
    # cache_abstraction(envs[0], params)

    if mode == 'train':
        train_agent_parallel(envs, params)
    elif mode == 'trainr':
        params['restore'] = restore
        train_agent_parallel(envs, params)
    elif mode == 'eval':
        params['restore'] = restore
        eval_agent_parallel(envs, params)
    else:
        print 'Unknown mode specified!'
        sys.exit(0)

if __name__ == '__main__':
    main()