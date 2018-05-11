#!/usr/bin/env python

# Python imports.
import sys
import numpy as np
from collections import defaultdict

# Other imports.
from simple_rl.agents import QLearningAgent
from simple_rl.tasks import FourRoomMDP
from simple_rl.planning import ValueIteration
from simple_rl.run_experiments import run_agents_on_mdp

# TODO:
# 1) Figure out rho 
# 2) 

# -------------------
# -- Entropy Funcs --
# -------------------

def kl(pmf_p, pmf_q):
    '''
    Args:
        pmf_p (dict)
        pmf_q (dict)

    Returns:
        (float)
    '''
    kl_divergence = 0.0
    for x in pmf_p.keys():
        if q[x] > 0.0:  # Avoid division by zero.
            kl_divergence += pmf_p[x] * math.log(pmf_p[x] / pmf_q[x])
    return d

def get_pmf_policy(policy, state_space, sample_rate=5):
    '''
    Args:
        policy (lambda): State ---> Action
        state (list)

    Returns:
        (dict)
    '''
    # Key = State, Val=dict:
        # Key = Action (str), Val = Prob (float).
    pmf_policy = defaultdict(lambda: defaultdict(float))

    for s_g in state_space:
        for i in xrange(sample_rate):
            pmf_policy[s_g][policy(s_g)] += 1.0 / sample_rate

    return pmf_policy

# -----------------------------
# -- Iterative BA Like Steps --
# -----------------------------

def compute_prob_of_s_phi(pmf_s, pmf_s_phi, coding_distr, beta):
    '''
    Args
    '''
    pass


def compute_coding_distr(ground_states, prev_phi, demo_policy, phi_policy, mdp, beta):
    '''
    Args:
        ground_states (list)
        prev_phi (dict)
        demo_policy (lambda)
        phi_policy (lambda)
        mdp (simple_rl.MDP)
        beta (float)

    Returns:
        (dict)

    Notes:

    '''
    # Get stationary abstract distribution.
    rho_phi = get_stationary_rho_from_policy(phi_policy, mdp)

    # Compute KL.
    pmf_demo_policy = get_pmf_policy(demo_policy, ground_states)
    abstract_states = [prev_phi(s) for s in ground_states]
    pmf_phi_policy = get_pmf_policy(phi_policy, abstract_states)


    

# ------------------------
# -- Init Distributions --
# ------------------------

def get_stationary_rho_from_policy(policy, mdp):
    '''
    Args:
        policy (dict): K=State ---> V=Pr(Action)
        mdp (simple_rl.MDP)

    Returns:
        (dict): policy (see above)
    '''
    pass

def init_identity_phi(num_ground_states):
	'''
	Args:
		num_ground_states (int)

	Returns:
		(float)
	'''
	new_coding_distr = defaultdict(lambda : defaultdict(float))

	# Initialize the identity distribution
	for s_g in xrange(num_ground_states):
		for s_phi in xrange(num_ground_states):
			new_coding_distr[s_g][s_phi] =  int(s_g == s_phi)

	return new_coding_distr

def init_uniform_phi(num_ground_states, num_abstr_states):
    '''
    Args:
        num_ground_states (int)
        num_abstr_states (int)

    Returns:
        (float)
    '''
    new_coding_distr = defaultdict(lambda : defaultdict(float))

    # Initialize the identity distribution
    for s_g in xrange(num_ground_states):
        for s_phi in xrange(num_abstr_states):
            new_coding_distr[s_g][s_phi] =  1.0 / num_abstr_states

    return new_coding_distr

# ----------------------------------
# -- Blahut Arimoto RL Main Steps --
# ----------------------------------

def main():
    # Setup MDP, Agents.
    mdp = FourRoomMDP(width=9, height=9, init_loc=(1, 1), goal_locs=[(9, 9)], gamma=0.95)

    # Make demonstrator policy.
    four_room_vi = ValueIteration(mdp)
    four_room_vi.run_vi()
    num_ground_states = four_room_vi.get_num_states()
    demonstrator_policy = four_room_vi.policy

    # Hyperparameter.
    beta = 1.0
    iters = 20

    # Init distributions. (stationary distributions)
    # TODO: init pmf s
    # TODO: init pmf s_phi
    coding_distr = init_identity_phi(num_ground_states)

    # Blahut.
    for i in range(iters):
        # TODO: implement compute_prob_of_s_phi
        pmf_s_phi = compute_prob_of_s_phi(pmf_s, coding_distr, beta=beta)
        
        # TODO: implement compute_coding_distr
        coding_distr = compute_coding_distr(pmf_s, pmf_s_phi, coding_distr, beta=beta)

    # Return the two distributions from BA.
    return pmf_s_phi, coding_distr

if __name__ == "__main__":
    main()
