#!/usr/bin/env python

# Python imports.
import sys
import math
import numpy as np
import random
from collections import defaultdict

# Other imports.
from simple_rl.abstraction.state_abs.ProbStateAbstractionClass import ProbStateAbstraction
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from simple_rl.agents import QLearningAgent, FixedPolicyAgent, RandomAgent
from simple_rl.tasks import FourRoomMDP, GridWorldMDP
from simple_rl.planning import ValueIteration
from simple_rl.run_experiments import run_agents_on_mdp
from blahut_arimoto import print_coding_distr, print_pmf, mutual_info
from rlit_utils import *

distance_func = kl

def get_pmf_policy(policy, state_space, actions, sample_rate=20):
    '''
    Args:
        policy (lambda): State ---> Action
        state_space (list)
        actions (list)

    Returns:
        (dict)
    '''
    # Key = State, Val=dict:
        # Key = Action (str), Val = Prob (float).
    pmf_policy = defaultdict(lambda: defaultdict(float))

    # Fill it with 0s first.
    for s in state_space:
        for a in actions:
            pmf_policy[s][a] = 0.0

    for s in state_space:
        for i in xrange(sample_rate):
            pmf_policy[s][policy(s)] += 1.0 / sample_rate

    return pmf_policy

def get_lambda_policy(policy):
    '''
    Args:
        policy (dict): K=State --> V=Dict, K=Action --> V=Probability

    Returns:
        (lambda)
    '''
    def pmf_policy(state):
        sampled_a_index = np.random.multinomial(1, policy[state].values()).tolist().index(1)
        return policy[state].keys()[sampled_a_index]

    return pmf_policy
# -----------------------------
# -- Iterative BA Like Steps --
# -----------------------------

def compute_prob_of_s_phi(pmf_s, coding_distr, ground_states, beta):
    '''
    Args
    '''
    new_pmf_s_phi = defaultdict(float)
    for s_phi in coding_distr.values()[0].keys():
        new_pmf_s_phi[s_phi] = sum([pmf_s[s] * coding_distr[s][s_phi] for s in ground_states])

    return new_pmf_s_phi

def _compute_denominator(s, pmf_s_phi, pi, abstr_pi, beta):
    '''
    Args:
        s
        pmf_s_phi
        pi
        abstr_pi
        beta

    Returns:
        (float)
    '''
    r = sum([pmf_s_phi[s_phi] * math.exp(-beta * distance_func(pi[s], abstr_pi[s_phi])) for s_phi in pmf_s_phi.keys()])
    return r

def compute_coding_distr(pmf_s, pmf_s_phi, demo_policy, abstr_policy, ground_states, actions, beta):
    '''
    Args:
        pmf_s (dict)
        pmf_s_phi (dict)
        demo_policy (lambda)
        ground_states (list)
        abstr_policy (lambda)
        beta (float)

    Returns:
        (dict)

    Notes:

    '''
    pmf_pi = get_pmf_policy(demo_policy, ground_states, actions)
    new_coding_distr = defaultdict(lambda: defaultdict(float))

    for s in ground_states:
        for s_phi in pmf_s_phi.keys():

            numerator = pmf_s_phi[s_phi] * math.exp(-beta * distance_func(pmf_pi[s], abstr_policy[s_phi]))
            denominator = _compute_denominator(s, pmf_s_phi, pmf_pi, abstr_policy, beta)

            # TODO: divide by zero?
            if denominator == 0:
                new_coding_distr[s][s_phi] = 0
            else:
                new_coding_distr[s][s_phi] = float(numerator) / denominator

    # TODO: KL is infinity everywhere initially. 
    # So: the first coding distribution we find sets 0 prob to all abstract states.

    return new_coding_distr

def compute_inv_coding_distr(pmf_s, pmf_s_phi, ground_states, coding_distr):
    inv_coding_distr = defaultdict(lambda: defaultdict(float))

    for s_phi in pmf_s_phi.keys():
        for s in ground_states:
            if pmf_s_phi[s_phi] == 0.0:
                inv_coding_distr[s_phi][s] = 0.0
            else:
                inv_coding_distr[s_phi][s] = coding_distr[s][s_phi] * pmf_s[s] / pmf_s_phi[s_phi]

    return inv_coding_distr


def compute_abstr_policy(demo_policy, ground_states, actions, coding_distr, pmf_s, pmf_s_phi):
    '''
    Args:
        demo_policy (lambda : s --> a)
        ground_states (list)
        actions (list)
        coding_distr (dict)

    Returns:
        (dict)
    '''
    pi = get_pmf_policy(demo_policy, ground_states, actions)
    abstr_pi = defaultdict(lambda: defaultdict(float))

    for s_phi in coding_distr[coding_distr.keys()[0]]:
        for a in actions:
            total = 0.0
            for s in ground_states:
                # TODO: Another case of divide by zero?
                if pmf_s_phi[s_phi] == 0:
                    continue
                total += pi[s][a] * coding_distr[s][s_phi] * (pmf_s[s] / pmf_s_phi[s_phi])
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
    '''
    Args:
        pmf
        actions (list)

    Returns:
        (dict)
    '''
    new_pi = defaultdict(lambda: defaultdict(float))

    for s in pmf.keys():
        for a in actions:
            new_pi[s][a] = 1.0 / len(actions)

    return new_pi

# ---------------------
# -- BARLEY Plotting --
# ---------------------

def make_barley_plot():
    '''
    Summary:
        Creates a plot showcasing the pmfs of the distributions
        computed by the blahut arimoto algorithm.
    '''
    from func_plotting import PlotFunc

    # Choose code-message combos to plot.
    funcs_to_plot = []

    # Set relevant params.
    mdp = GridWorldMDP(width=11, height=11, init_loc=(1, 1), goal_locs=[(11, 11)], gamma=0.95, slip_prob=0.05)
    param_dict = {"mdp":mdp, "iters":100, "convergence_threshold":0.001}

    # Make plot func object.
    plot_beta_vs_s_phi_size = PlotFunc(_barley_s_phi_size_plot_wrapper, param_dict=param_dict, x_min=0.0, x_max=100.0, x_interval=10)

    # Hang on to it.
    funcs_to_plot.append(plot_beta_vs_s_phi_size)

    # Plot.
    from func_plotting import plot_funcs
    plot_funcs(funcs_to_plot, title="BARLEY: $\\beta$  vs. $|S_\\phi|$", x_label="$\\beta$", y_label="$|S_\\phi|$")

def _barley_s_phi_size_plot_wrapper(x, param_dict):
    '''
    Args:
        x (float): stands for $\beta$ in the BARLEY algorithm.
        param_dict (dict): contains relevant parameters for plotting.

    Returns:
        (int): The size of the computed abstract state space.

    Notes:
        This serves as a wrapper to cooperate with PlotFunc.
    '''
    mdp = param_dict["mdp"]
    iters = param_dict["iters"]
    convergence_threshold = param_dict["convergence_threshold"]

    pmf_code, coding_distr, abstr_pi = barley(mdp, beta=x, iters=iters, convergence_threshold=convergence_threshold)

    s_phi_size = get_sa_size_from_coding_distr(coding_distr)

    return s_phi_size

# ----------------------------------
# -- Blahut Arimoto RL Main Steps --
# ----------------------------------

def barley(mdp, iters=100, beta=5.0, convergence_threshold=0.0001):
    '''
    Args:
        mdp (simple_rl.MDP)
        iters (int)
        beta (float)
        convergence_threshold (float): When KL(phi_{t+1}, phi_t) < @convergence_threshold, we stop iterating.

    Returns:
        (dict): P(s_phi)
        (dict): P(s_phi | s)
        (dict): P(a | s_phi)

    Summary:
        Runs the Blahut-Arimoto like algorithm for the given mdp.
    '''
    # Make demonstrator policy.
    demo_vi = ValueIteration(mdp)
    demo_vi.run_vi()
    ground_states = demo_vi.get_states()
    actions = mdp.get_actions()
    num_ground_states = demo_vi.get_num_states()
    demo_policy = demo_vi.policy

    # Init distributions. (stationary distributions)
    pmf_s = get_stationary_rho_from_policy(demo_policy, mdp)
    coding_distr = init_identity_phi(ground_states)
    pmf_s_phi = compute_prob_of_s_phi(pmf_s, coding_distr, ground_states, beta=beta)
    abstr_pi = init_uniform_pi(pmf_s_phi, actions)

    # Blahut.
    for i in range(iters):
        print 'Iteration {0} of {1}'.format(i+1, iters)

        # (A) Compute \rho(s).
        pmf_s_phi = compute_prob_of_s_phi(pmf_s, coding_distr, ground_states, beta=beta)

        # (B) Compute \phi.
        next_coding_distr = compute_coding_distr(pmf_s, pmf_s_phi, demo_policy, abstr_pi, ground_states, actions, beta=beta)

        print next_coding_distr

        # (C) Compute \pi_\phi.
        abstr_pi = compute_abstr_policy(demo_policy, ground_states, actions, next_coding_distr, pmf_s, pmf_s_phi)

        print abstr_pi

        # Check if we're converged.
        if max([distance_func(next_coding_distr[s], coding_distr[s]) for s in ground_states]) < convergence_threshold:
            break
        
        # Iterate.
        coding_distr = next_coding_distr

    return pmf_s_phi, coding_distr, abstr_pi

def get_sa_size_from_coding_distr(coding_distr):
    '''
    Args:
        coding_distr (dict)

    Returns:
        (int)
    '''
    # Print out coding distr.
    total = 0
    s_phi_set = set([])
    for s in coding_distr:
        for s_phi in coding_distr[s]:
            if coding_distr[s][s_phi] > 0 and s_phi not in s_phi_set:
                s_phi_set.add(s_phi)

    return len(s_phi_set)

def barley_compare_policies():
    # Make MDP.
    # mdp = GridWorldMDP(width=2, height=3, init_loc=(1, 1), goal_locs=[(2, 3)], gamma=0.95)
    mdp = GridWorldMDP(width=10, height=10, init_loc=(1, 1), goal_locs=[(9, 9)], gamma=0.95)

    # Run BARLEY.
    pmf_s_phi, coding_distr, abstr_pi = barley(mdp)

    # Make demonstrator policy.
    demo_vi = ValueIteration(mdp)
    demo_vi.run_vi()
    demo_policy = demo_vi.policy

    demo_agent = FixedPolicyAgent(demo_policy, name="$\\pi_d$")

    # Make abstract agents.
    lambda_abstr_policy = get_lambda_policy(abstr_pi)
    prob_s_phi = ProbStateAbstraction(coding_distr)
    abstr_agent = AbstractionWrapper(FixedPolicyAgent, state_abstr=prob_s_phi, agent_params={"policy":lambda_abstr_policy, "name":"$\\pi_\\phi$"}, name_ext="")
    rand_agent = RandomAgent(mdp.get_actions())

    # Run.
    run_agents_on_mdp([demo_agent, abstr_agent, rand_agent], mdp, episodes=1, steps=1000)

    # Print state space sizes.
    abstr_state_space_size = get_sa_size_from_coding_distr(coding_distr)
    print "\nState Spaces Sizes:"
    print "\t|S_\\phi| =", abstr_state_space_size
    print "\t|S| =", len(demo_vi.get_states()), "\n\n"

def main():

    exp_type = "compare_policies"

    if exp_type == "beta_plot":
        make_barley_plot()
    elif exp_type == "compare_policies":
        barley_compare_policies()

if __name__ == "__main__":
    main()
