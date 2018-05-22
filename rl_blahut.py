#!/usr/bin/env python

# Python imports.
import sys
import math
import numpy as np
from collections import defaultdict

# Other imports.
from simple_rl.agents import QLearningAgent
from simple_rl.tasks import ChainMDP
from simple_rl.tasks import FourRoomMDP
from simple_rl.planning import ValueIteration
from simple_rl.run_experiments import run_agents_on_mdp
from blahut_arimoto import print_coding_distr

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
        if pmf_q[x] > 0.0:  # Avoid division by zero.
            if pmf_p[x] == 0.0:
                return float('inf')
            kl_divergence += pmf_p[x] * math.log(pmf_p[x] / pmf_q[x])
    return kl_divergence

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

def compute_prob_of_s_phi(pmf_s, coding_distr, beta):
    '''
    Args
    '''
    new_pmf_s_phi = defaultdict(float)
    for s_phi in coding_distr.values()[0].keys():
        new_pmf_s_phi[s_phi] = sum([pmf_s[s] * coding_distr[s][s_phi] for s in pmf_s.keys()])

    return new_pmf_s_phi

def _compute_denominator(s, pmf_s_phi, pi, abstr_pi, beta):
    '''
    '''
    return sum([pmf_s_phi[s_phi] * math.exp(-beta * kl(pi[s], abstr_pi[s_phi])) for s_phi in pmf_s_phi.keys()])

def compute_coding_distr(pmf_s, pmf_s_phi, demonstrator_policy, ground_states, abstr_pi, beta):
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
    pi = get_pmf_policy(demonstrator_policy, ground_states)
    new_coding_distr = defaultdict(lambda: defaultdict(float))

    for s in pmf_s.keys():
        for s_phi in pmf_s_phi.keys():
            numerator = pmf_s_phi[s_phi] * math.exp(-beta * kl(pi[s], abstr_pi[s_phi]))
            denominator = _compute_denominator(s, pmf_s_phi, pi, abstr_pi, beta)

            new_coding_distr[s][s_phi] = float(numerator) / denominator

    return new_coding_distr


def compute_inv_coding_distr(pmf_s, pmf_s_phi, coding_distr):
    inv_coding_distr = defaultdict(lambda: defaultdict(float))

    for s_phi in pmf_s_phi.keys():
        for s in pmf_s.keys():
            if pmf_s_phi[s_phi] == 0.0:
                inv_coding_distr[s_phi][s] = 0.0
            else:
                inv_coding_distr[s_phi][s] = coding_distr[s][s_phi] * pmf_s[s] / pmf_s_phi[s_phi]

    return inv_coding_distr


def compute_abstr_policy(demonstrator_policy, ground_states, actions, inv_coding_distr):
    pi = get_pmf_policy(demonstrator_policy, ground_states)
    abstr_pi = defaultdict(lambda: defaultdict(float))

    for s_phi in inv_coding_distr.keys():
        for a in actions:
            total = 0.0
            for s in ground_states:
                total += pi[s][a] * inv_coding_distr[s_phi][s]
            abstr_pi[s_phi][a] = total

    return abstr_pi

# ------------------------
# -- Init Distributions --
# ------------------------

def get_stationary_rho_from_policy(policy, mdp, sample_rate=10, max_steps=30):
    '''
    Args:
        policy (dict): K=State ---> V=Pr(Action)
        mdp (simple_rl.MDP)

    Returns:
        (dict): policy (see above)
    '''

    s = mdp.get_init_state()
    rho = defaultdict(float)
    total = 0
    for _ in range(sample_rate):
        num_steps = 0
        while not s.is_terminal() and num_steps < max_steps:
            rho[s] += 1.0
            total += 1
            _, s = mdp.execute_agent_action(policy(s))
            num_steps += 1

        if s.is_terminal():
            rho[s] += 1.0
            total += 1

        mdp.reset()
        s = mdp.get_init_state()

    for k in rho.keys():
        rho[k] = rho[k] / total

    return rho

def init_identity_phi(ground_states):
	'''
	Args:
		num_ground_states (int)

	Returns:
		(float)
	'''
	new_coding_distr = defaultdict(lambda : defaultdict(float))

	# Initialize the identity distribution
	for id, s_g in enumerate(ground_states):
		for s_phi in xrange(len(ground_states)):
			new_coding_distr[s_g][s_phi] =  int(id == s_phi)

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

def init_uniform_pi(pmf, actions):
    new_pi = defaultdict(lambda: defaultdict(float))

    for s in pmf.keys():
        for a in actions:
            new_pi[s][a] = 1.0 / len(actions)

    return new_pi

# ----------------------------------
# -- Blahut Arimoto RL Main Steps --
# ----------------------------------

def main():
    # Setup MDP, Agents.
    mdp = FourRoomMDP(width=5, height=5, init_loc=(1, 1), goal_locs=[(5, 5)], gamma=0.95)

    # Make demonstrator policy.
    four_room_vi = ValueIteration(mdp)
    four_room_vi.run_vi()
    ground_states = four_room_vi.get_states()
    actions = mdp.get_actions()
    num_ground_states = four_room_vi.get_num_states()
    demonstrator_policy = four_room_vi.policy

    # Hyperparameters.
    beta = 100.0
    iters = 40

    # Init distributions. (stationary distributions)
    pmf_s = get_stationary_rho_from_policy(demonstrator_policy, mdp)
    coding_distr = init_identity_phi(ground_states)
    pmf_s_phi = compute_prob_of_s_phi(pmf_s, coding_distr, beta=beta)
    abstr_pi = init_uniform_pi(pmf_s_phi, actions)

    # Blahut.
    for i in range(iters):
        print 'Iteration {0} of {1}'.format(i+1, iters)

        # (A) Compute \rho(s).
        pmf_s_phi = compute_prob_of_s_phi(pmf_s, coding_distr, beta=beta)

        # (B) Compute \phi.
        coding_distr = compute_coding_distr(pmf_s, pmf_s_phi, demonstrator_policy, ground_states, abstr_pi, beta=beta)

        # (C) Compute \pi_\phi.
        inv_coding_distr = compute_inv_coding_distr(pmf_s, pmf_s_phi, coding_distr)
        abstr_pi = compute_abstr_policy(demonstrator_policy, ground_states, actions, inv_coding_distr)

    print_coding_distr(coding_distr)
    # for k, v in coding_distr.iteritems():
    #     print k , 

    return pmf_s_phi, coding_distr, abstr_pi

if __name__ == "__main__":
    main()
