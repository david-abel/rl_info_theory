#!/usr/bin/env python

# Python imports.
import sys
import math
import numpy as np

# Other imports.
from simple_rl.tasks import FourRoomMDP
from simple_rl.utils import make_mdp
from simple_rl.mdp import State
from simple_rl.agents import QLearningAgent
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
from simple_rl.planning import ValueIteration
from simple_rl.run_experiments import run_agents_lifelong
from simple_rl.abstraction.AbstractValueIterationClass import AbstractValueIteration
from simple_rl.run_experiments import evaluate_agent
from rlit_utils import *
from info_sa import *

def make_lifelong_sa_info_sa(mdp_distr, beta):
    '''
    Args:
        mdp_distr (simple_rl.MDPDistribution)
        beta (float)

    Returns:
        (simple_rl.StateAbstraction)
    '''

    master_sa = None
    for mdp in mdp_distr.get_all_mdps():

        # Get demo policy.
        vi = ValueIteration(mdp)
        vi.run_vi()
        demo_policy = get_lambda_policy(make_det_policy_eps_greedy(vi.policy, vi.get_states(), mdp.get_actions(), epsilon=0.2))

        # Get abstraction.
        pmf_s_phi, phi_pmf, abstr_policy_pmf = run_info_sa(mdp, demo_policy, beta=beta)

        crisp_sa = convert_prob_sa_to_sa(ProbStateAbstraction(phi_pmf))

        if master_sa is None:
            master_sa = crisp_sa
        else:
            master_sa = master_sa + crisp_sa

    return master_sa

class SAVI(object):
    def __init__(self, sa, abstr_policy):
        self.sa = sa
        self.abstr_policy = abstr_policy

    def policy(self, state):
        return self.abstr_policy(self.sa.phi(state)).act(state)

def evaluate_lifelong_sa(lifelong_sa, mdp_distr, samples=10):
    '''
    Args:
        lifelong_sa (simple_rl.abstraction.StateAbstraction)
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
        # abstr_mdp = make_abstr_mdp(mdp, state_abstr=lifelong_sa)
        # abstr_vi = ValueIteration(abstr_mdp, sample_rate=20)
        abstr_vi = AbstractValueIteration(mdp, state_abstr=lifelong_sa)
        abstr_vi.run_vi()
        abstr_opt_policy_mapper = SAVI(lifelong_sa, abstr_vi.policy)
        abstr_opt_agent = FixedPolicyAgent(abstr_opt_policy_mapper.policy, "abstract")

        # Compare.
        avg_opt_val += evaluate_agent(opt_agent, mdp) / samples
        avg_abstr_opt_val += evaluate_agent(abstr_opt_agent, mdp) / samples

    print "Ground:", lifelong_sa.get_num_ground_states(), round(avg_opt_val, 4)
    print "Abstract:", lifelong_sa.get_num_abstr_states(), round(avg_abstr_opt_val, 4)
    print

def main():

    # Make MDP Distribution.
    mdp_distr = make_mdp.make_mdp_distr(mdp_class="four_room", grid_dim=9)

    # Make SA.
    lifelong_sa = make_lifelong_sa_info_sa(mdp_distr, beta=100.0)

    # Make agent.
    # ql_agent = QLearningAgent(mdp_distr.get_actions())
    # abstr_ql = AbstractionWrapper(QLearningAgent, state_abstr=lifelong_sa, agent_params={"actions":mdp_distr.get_actions()})
    # run_agents_lifelong([abstr_ql, ql_agent], mdp_distr, steps=200, samples=20, episodes=100)

    # Visualize.
    from simple_rl.abstraction.state_abs.sa_helpers import visualize_state_abstr_grid
    visualize_state_abstr_grid(mdp_distr.sample(), lifelong_sa)

    # Evaluate.
    # evaluate_lifelong_sa(lifelong_sa, mdp_distr)
    
if __name__ == "__main__":
    main()
