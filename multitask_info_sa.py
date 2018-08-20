#!/usr/bin/env python

# Python imports.
import sys
import math
import numpy as np
import itertools

# Other imports.
from simple_rl.tasks import FourRoomMDP
from simple_rl.mdp import State
from simple_rl.agents import QLearningAgent
from simple_rl.run_experiments import run_agents_lifelong
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
from simple_rl.abstraction.state_abs import sa_helpers
from simple_rl.planning import ValueIteration
from simple_rl.abstraction.AbstractValueIterationClass import AbstractValueIteration
from make_mdp_distribution import make_mdp_distr
from rlit_utils import *
from info_sa import *

def make_multitask_sa_info_sa(mdp_distr, beta, is_deterministic_ib=False):
    '''
    Args:
        mdp_distr (simple_rl.MDPDistribution)
        beta (float)
        is_deterministic_ib (float)

    Returns:
        (simple_rl.StateAbstraction)
    '''

    master_sa = None
    all_state_absr = []
    for mdp in mdp_distr.get_all_mdps():

        # Get demo policy.
        vi = ValueIteration(mdp)
        vi.run_vi()
        demo_policy = get_lambda_policy(make_det_policy_eps_greedy(vi.policy, vi.get_states(), mdp.get_actions(), epsilon=0.2))

        # Get abstraction.
        pmf_s_phi, phi_pmf, abstr_policy_pmf = run_info_sa(mdp, demo_policy, beta=beta, is_deterministic_ib=is_deterministic_ib)
        crisp_sa = convert_prob_sa_to_sa(ProbStateAbstraction(phi_pmf))
        all_state_absr.append(crisp_sa)

    # Make master state abstr by intersection.
    vi = ValueIteration(mdp_distr.get_all_mdps()[0])
    ground_states = vi.get_states()

    master_sa = sa_helpers.merge_state_abstr(all_state_absr, ground_states)

    return master_sa


class SAVI(object):
    def __init__(self, sa, abstr_policy):
        self.sa = sa
        self.abstr_policy = abstr_policy

    def policy(self, state):
        return self.abstr_policy(self.sa.phi(state)).act(state)

def evaluate_multitask_sa(multitask_sa, mdp_distr, samples=10):
    '''
    Args:
        multitask_sa (simple_rl.abstraction.StateAbstraction)
        mdp_distr (simple_rl.mdp.MDPDistribution)
        samples (float)
    '''

    # Average value over @samples.
    avg_opt_val = 0.0
    avg_abstr_opt_val = 0.0
    for i in range(samples):
        mdp = mdp_distr.sample()
        
        # Optimal Policy.
        vi = ValueIteration(mdp)
        vi.run_vi()
        opt_agent = FixedPolicyAgent(vi.policy)

        # Evaluate Optimal Abstract Policy.
        # abstr_mdp = make_abstr_mdp(mdp, state_abstr=multitask_sa)
        # abstr_vi = ValueIteration(abstr_mdp, sample_rate=20)
        abstr_vi = AbstractValueIteration(mdp, state_abstr=multitask_sa)
        abstr_vi.run_vi()
        abstr_opt_policy_mapper = SAVI(multitask_sa, abstr_vi.policy)
        abstr_opt_agent = FixedPolicyAgent(abstr_opt_policy_mapper.policy, "abstract")

        # Compare.
        avg_opt_val += evaluate_agent(opt_agent, mdp) / samples
        avg_abstr_opt_val += evaluate_agent(abstr_opt_agent, mdp) / samples

    print "Ground:", multitask_sa.get_num_ground_states(), round(avg_opt_val, 4)
    print "Abstract:", multitask_sa.get_num_abstr_states(), round(avg_abstr_opt_val, 4)
    print

def main():

    # Make MDP Distribution.
    mdp_distr = make_mdp_distr(mdp_class="four_room", grid_dim=11, slip_prob=0.05, gamma=0.99)

    # Make SA.
    multitask_sa_beta_1 = make_multitask_sa_info_sa(mdp_distr, beta=1.0, is_deterministic_ib=True)
    multitask_sa_beta_10 = make_multitask_sa_info_sa(mdp_distr, beta=10.0, is_deterministic_ib=True)
    multitask_sa_beta_100 = make_multitask_sa_info_sa(mdp_distr, beta=100.0, is_deterministic_ib=True)
    multitask_sa_beta_1000 = make_multitask_sa_info_sa(mdp_distr, beta=1000.0, is_deterministic_ib=True)

    # Make agent.
    ql_agent = QLearningAgent(mdp_distr.get_actions())
    abstr_ql_b1 = AbstractionWrapper(QLearningAgent, state_abstr=multitask_sa_beta_1, agent_params={"actions":mdp_distr.get_actions()}, name_ext="-$\\phi_{\\beta = 1}$")
    abstr_ql_b10 = AbstractionWrapper(QLearningAgent, state_abstr=multitask_sa_beta_10, agent_params={"actions":mdp_distr.get_actions()}, name_ext="-$\\phi_{\\beta = 10}$")
    abstr_ql_b100 = AbstractionWrapper(QLearningAgent, state_abstr=multitask_sa_beta_100, agent_params={"actions":mdp_distr.get_actions()}, name_ext="-$\\phi_{\\beta = 100}$")
    abstr_ql_b1000 = AbstractionWrapper(QLearningAgent, state_abstr=multitask_sa_beta_1000, agent_params={"actions":mdp_distr.get_actions()}, name_ext="-$\\phi_{\\beta = 1000}$")
    run_agents_lifelong([abstr_ql_b1, abstr_ql_b10, abstr_ql_b100, abstr_ql_b1000, ql_agent], mdp_distr, steps=200, samples=50, episodes=200)

    # Visualize.
    # from simple_rl.abstraction.state_abs.sa_helpers import visualize_state_abstr_grid
    # visualize_state_abstr_grid(mdp_distr.sample(), multitask_sa)

    # Evaluate.
    # evaluate_multitask_sa(multitask_sa, mdp_distr)
    
if __name__ == "__main__":
    main()
