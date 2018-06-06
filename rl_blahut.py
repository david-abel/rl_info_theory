#!/usr/bin/env python

# Python imports.
import sys
import math
import numpy as np
import random
from collections import defaultdict

# Other imports.
from simple_rl.abstraction.state_abs.ProbStateAbstractionClass import ProbStateAbstraction, convert_prob_sa_to_sa
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from simple_rl.agents import QLearningAgent, FixedPolicyAgent, RandomAgent
from simple_rl.tasks import FourRoomMDP, GridWorldMDP, ChainMDP
from simple_rl.mdp import State
from simple_rl.planning import ValueIteration
from simple_rl.run_experiments import run_agents_on_mdp, evaluate_agent
from blahut_arimoto import print_coding_distr, print_pmf, mutual_info
from rlit_utils import *

distance_func = l1_distance

# ----------------------
# -- Policy Utilities --
# ----------------------

def get_pmf_policy(policy, state_space, actions, sample_rate=30):
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
    def lambda_policy(state):
        sampled_a_index = np.random.multinomial(1, policy[state].values()).tolist().index(1)
        return policy[state].keys()[sampled_a_index]

    return lambda_policy

# -----------------------------
# -- Iterative BA Like Steps --
# -----------------------------

def compute_prob_of_s_phi(pmf_s, coding_distr, ground_states, abstract_states, beta):
    '''
    Args:
        pmf_s (dict)
        coding_distr (dict)
        ground_states (list)
        abstract_states (list)
        beta (float)

    Returns:
        (dict)
    '''
    new_pmf_s_phi = defaultdict(float)
    for s_phi in abstract_states:
        new_pmf_s_phi[s_phi] = sum([pmf_s[s] * coding_distr[s][s_phi] for s in ground_states])

    return new_pmf_s_phi

def _compute_denominator(s, pmf_s_phi, demo_policy, abstr_policy, beta):
    '''
    Args:
        s (simple_rl.State)
        pmf_s_phi (dict)
        demo_policy (dict)
        abstr_policy (dict)
        beta (float)

    Returns:
        (float)
    '''
    return sum([pmf_s_phi[s_phi] * math.exp(-beta * distance_func(demo_policy[s], abstr_policy[s_phi])) for s_phi in pmf_s_phi.keys()])

def compute_coding_distr(pmf_s, pmf_s_phi, demo_policy, abstr_policy, ground_states, abstract_states, beta):
    '''
    Args:
        pmf_s (dict)
        pmf_s_phi (dict)
        demo_policy (dict)
        abstr_policy (dict)
        abstract_states (list)
        ground_states (list)
        beta (float)

    Returns:
        (dict)

    '''
    new_coding_distr = defaultdict(lambda: defaultdict(float))

    # The problem: for a given s, numerator is always the same for each s-sphi pair.

    for s in ground_states:
        for s_phi in abstract_states:
            numerator = pmf_s_phi[s_phi] * math.exp(-beta * distance_func(demo_policy[s], abstr_policy[s_phi]))
            denominator = _compute_denominator(s, pmf_s_phi, demo_policy, abstr_policy, beta)

            if denominator == 0:
                print "Warning: division by zero in compute_coding_distr."
                new_coding_distr[s][s_phi] = 0
            else:    
                new_coding_distr[s][s_phi] = numerator / denominator

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
    abstr_policy = defaultdict(lambda: defaultdict(float))

    for s_phi in pmf_s_phi.keys():
        for a in actions:

            total = 0.0
            for s in ground_states:
                if pmf_s_phi[s_phi] == 0:
                    # Avoid division by zero.
                    continue

                total += demo_policy[s][a] * coding_distr[s][s_phi] * (pmf_s[s] / pmf_s_phi[s_phi])
            
            abstr_policy[s_phi][a] = total

    return abstr_policy

# ------------------------
# -- Init Distributions --
# ------------------------

def get_stationary_rho_from_policy(policy, mdp, ground_states, sample_rate=10, max_steps=30):
    '''
    Args:
        policy (dict): K=State ---> V=Pr(Action)
        mdp (simple_rl.MDP)
        ground_states (list)

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

    for k in ground_states:
        rho[k] = rho[k] / total

    return rho

def init_random_phi(ground_states):
    '''
    Args:
        num_ground_states (int)

    Returns:
        (float)
    '''
    new_coding_distr = defaultdict(lambda : defaultdict(float))

    # Initialize the identity distribution
    for id, s_g in enumerate(ground_states):
        s_phi_map = random.choice(range(len(ground_states)))
        for s_phi in xrange(len(ground_states)):
            new_coding_distr[s_g][State(s_phi)] = int(s_phi_map == s_phi)

    return new_coding_distr

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
			new_coding_distr[s_g][State(s_phi)] =  int(id == s_phi)

	return new_coding_distr

def init_uniform_phi(num_ground_states):
    '''
    Args:
        num_ground_states (int)

    Returns:
        (float)
    '''
    new_coding_distr = defaultdict(lambda : defaultdict(float))

    # Initialize the identity distribution
    for s_g in range(num_ground_states):
        for s_phi in range(num_ground_states):
            new_coding_distr[s_g][State(s_phi)] =  1.0 / num_ground_states

    return new_coding_distr

def init_random_rho_phi(ground_states):
    '''
    Args:
        ground_states (list)

    Returns:
        (dict)
    '''
    new_rho_phi = defaultdict(float)
    state_distr = np.random.dirichlet([0.5] * len(ground_states)).tolist()
    for i in range(len(ground_states)):
        new_rho_phi[State(i)] = state_distr[i]

    return new_rho_phi

def init_uniform_rho_phi(ground_states):
    '''
    Args:
        ground_states (list)

    Returns:
        (dict)
    '''
    new_rho_phi = defaultdict(float)
    for i in range(len(ground_states)):
        new_rho_phi[i] = 1.0 / len(ground_states)

    return new_rho_phi

def init_random_pi(pmf, actions):
    '''
    Args:
        pmf
        actions (list)

    Returns:
        (dict)
    '''
    new_pi = defaultdict(lambda: defaultdict(float))

    for s in pmf.keys():
        action_distr = np.random.dirichlet([0.5] * len(actions)).tolist()
        for i, a in enumerate(actions):
            new_pi[s][a] = action_distr[i]

    return new_pi

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


# -----------------------
# -- BARLEY Main Steps --
# -----------------------

def barley(mdp, iters=100, beta=20.0, convergence_threshold=0.01):
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
    print "~"*16
    print "~~ BETA =", beta, "~~"
    print "~"*16
    # Make demonstrator policy.
    demo_vi = ValueIteration(mdp)
    demo_vi.run_vi()
    ground_states = demo_vi.get_states()
    actions = mdp.get_actions()
    num_ground_states = demo_vi.get_num_states()
    demo_policy = make_det_policy_eps_greedy(demo_vi.policy, demo_vi.get_states(), mdp.get_actions())
    demo_lambda_policy = get_lambda_policy(demo_policy)

    # Init distributions. (stationary distributions)
    pmf_s = get_stationary_rho_from_policy(demo_lambda_policy, mdp, ground_states)
    pmf_s_phi = init_random_rho_phi(ground_states)
    coding_distr = init_random_phi(ground_states)
    abstr_policy = init_random_pi(pmf_s_phi, actions)

    # Abstract state space.
    abstract_states = pmf_s_phi.keys()

    # BARLEY.
    for i in range(iters):
        print 'Iteration {0} of {1}'.format(i+1, iters)

        # (A) Compute \phi.
        next_coding_distr = compute_coding_distr(pmf_s, pmf_s_phi, demo_policy, abstr_policy, ground_states, abstract_states, beta=beta)

        # (B) Compute \rho(s).
        next_pmf_s_phi = compute_prob_of_s_phi(pmf_s, next_coding_distr, ground_states, abstract_states, beta=beta)

        # (C) Compute \pi_\phi.
        next_abstr_policy = compute_abstr_policy(demo_policy, ground_states, actions, next_coding_distr, pmf_s, next_pmf_s_phi)

        # Convergence checks.
        coding_update_delta = max([l1_distance(next_coding_distr[s], coding_distr[s]) for s in ground_states])
        policy_update_delta = max([l1_distance(next_abstr_policy[s_phi], abstr_policy[s_phi]) for s_phi in abstract_states])
        state_distr_update_delta = l1_distance(next_pmf_s_phi, pmf_s_phi)

        # Debugging.
        is_coding_converged = coding_update_delta < convergence_threshold
        is_policy_converged = policy_update_delta < convergence_threshold
        is_pmf_s_phi_converged = state_distr_update_delta < convergence_threshold

        # Keep:

        # Update pointers.
        coding_distr = next_coding_distr
        pmf_s_phi = next_pmf_s_phi
        abstr_policy = next_abstr_policy
        # print "c, p, s:", round(coding_update_delta, 4), round(policy_update_delta, 4), round(state_distr_update_delta, 4)

        if is_coding_converged and is_policy_converged and is_pmf_s_phi_converged:
            print "\tBARLEY Converged."
            break

    return pmf_s_phi, coding_distr, abstr_policy

def get_sa_size_from_coding_distr(coding_distr):
    '''
    Args:
        coding_distr (dict)

    Returns:
        (int)
    '''
    total = 0
    s_phi_set = set([])
    for s in coding_distr:
        for s_phi in coding_distr[s]:
            if coding_distr[s][s_phi] > 0 and s_phi not in s_phi_set:
                s_phi_set.add(s_phi)

    return len(s_phi_set)

def make_det_policy_eps_greedy(lambda_policy, ground_states, actions, epsilon=0.1):
    '''
    Args:
        lambda_policy (lambda: state --> action)
        ground_states (list)
        actions (list)
        epsilon (float)

    Returns:
        (dict)
    '''
    pmf_policy = defaultdict(lambda: defaultdict(float))

    for s in ground_states:
        for a in actions:
            if a == lambda_policy(s):
                pmf_policy[s][a] = 1.0 - epsilon
            else:
                pmf_policy[s][a] = epsilon / (len(actions) - 1)

    return pmf_policy

# ---------------------
# -- BARLEY Plotting --
# ---------------------

def make_barley_plot(plotting_helper_func, file_name="barley", x_label="$\\beta$", y_label="$|S_\\phi|$"):
    '''
    Args:
        plotting_helper_func (lambda : _ --> numeric)
        file_name (str)
        x_label (str)
        y_label (str)

    Summary:
        Creates a plot showcasing the pmfs of the distributions
        computed by the blahut arimoto algorithm.
    '''
    from func_plotting import PlotFunc

    # Set relevant params.
    grid_dim = 5
    mdp = FourRoomMDP(width=grid_dim, height=grid_dim, init_loc=(1, 1), goal_locs=[(grid_dim, grid_dim)], gamma=0.9)
    # mdp = GridWorldMDP(width=5, height=5, init_loc=(1, 1), goal_locs=[(3, 3)], gamma=0.9)
    param_dict = {"mdp":mdp, "iters":200, "convergence_threshold":0.0001}

    # Make plot func object.
    x_min, x_max, x_interval = 0.0, grid_dim, grid_dim / 20.0

    if "|S" in y_label:
        # Show ground state space size.
        vi = ValueIteration(mdp)
        num_ground_states = len(vi.get_states())
        def plot_func_num_ground_states(x, param_dict):
            return num_ground_states
        ground_plot = PlotFunc(plot_func_num_ground_states, series_name="$|S|$", param_dict={}, x_min=x_min, x_max=x_max, x_interval=x_interval)

        regular_series_name = "$|S_{\\phi-\\Pr}|$"
        crispy_series_name = "$|S_{\\phi}|$"

    elif "V" in y_label:
        # Show demo policy value.
        vi = ValueIteration(mdp)
        vi.run_vi()
        demo_agent = FixedPolicyAgent(vi.policy)
        val = evaluate_agent(demo_agent, mdp, instances=10)

        def plot_func_demo_pol_value(x, param_dict):
            return val

        regular_series_name = "$\\pi^{\\phi-\\Pr}$"
        crispy_series_name = "$\\pi^{\\phi}$"

        ground_plot = PlotFunc(plot_func_demo_pol_value, series_name="$\\pi_d$", param_dict={}, x_min=x_min, x_max=x_max, x_interval=x_interval)


    # Make barley plots.
    barley_plot_beta_vs = PlotFunc(plotting_helper_func, series_name=regular_series_name, param_dict=dict(param_dict.items() + {"use_crisp_sa":False}.items()), x_min=x_min, x_max=x_max, x_interval=x_interval)
    barley_plot_beta_vs_crisp = PlotFunc(plotting_helper_func, series_name=crispy_series_name, param_dict=dict(param_dict.items() + {"use_crisp_sa":True}.items()), x_min=x_min, x_max=x_max, x_interval=x_interval)

    # Plot.
    from func_plotting import plot_funcs
    plot_funcs([barley_plot_beta_vs, barley_plot_beta_vs_crisp, ground_plot], file_name=file_name, title="BARLEY: " + x_label + " vs. " + y_label, x_label=x_label, y_label=y_label, use_legend=True)

def _barley_val_plot_wrapper(x, param_dict):
    '''
    Args:
        x (float): stands for $\beta$ in the BARLEY algorithm.
        param_dict (dict): contains relevant parameters for plotting.

    Returns:
        (int): The value achieved by \pi_\phi^* in the MDP.

    Notes:
        This serves as a wrapper to cooperate with PlotFunc.
    '''
    mdp = param_dict["mdp"]
    iters = param_dict["iters"]
    convergence_threshold = param_dict["convergence_threshold"]
    use_crisp_sa = param_dict["use_crisp_sa"]

    pmf_code, coding_distr, abstr_policy = barley(mdp, beta=x, iters=iters, convergence_threshold=convergence_threshold)

    # Make abstract agent.
    lambda_abstr_policy = get_lambda_policy(abstr_policy)
    prob_s_phi = ProbStateAbstraction(coding_distr)

    # Convert to crisp SA if needed.
    phi = convert_prob_sa_to_sa(prob_s_phi) if use_crisp_sa else prob_s_phi

    abstr_agent = AbstractionWrapper(FixedPolicyAgent, state_abstr=phi, agent_params={"policy":lambda_abstr_policy, "name":"$\\pi_\\phi$"}, name_ext="")
    
    # Compute value of abstract policy w/ coding distribution.
    value = evaluate_agent(agent=abstr_agent, mdp=mdp, instances=1000)

    print "v", value

    return value


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
    # 
    mdp = param_dict["mdp"]
    iters = param_dict["iters"]
    convergence_threshold = param_dict["convergence_threshold"]
    use_crisp_sa = param_dict["use_crisp_sa"]

    # Let BARLEY run.
    pmf_code, coding_distr, abstr_policy = barley(mdp, beta=x, iters=iters, convergence_threshold=convergence_threshold)

    # Check size.
    prob_s_phi = ProbStateAbstraction(coding_distr)
    if use_crisp_sa:
        phi = convert_prob_sa_to_sa(prob_s_phi)
        s_phi_size = phi.get_num_abstr_states()
    else:
        s_phi_size = get_sa_size_from_coding_distr(coding_distr)

    return s_phi_size

# ---------------------
# -- Main Experiment --
# ---------------------

def barley_compare_policies():
    # Make MDP.
    mdp = FourRoomMDP(width=5, height=5, init_loc=(1, 1), goal_locs=[(5, 5)], gamma=0.9)

    # Run BARLEY.
    pmf_s_phi, coding_distr, abstr_policy = barley(mdp, iters=100, beta=3, convergence_threshold=0.00001)

    # Make demonstrator policy.
    demo_vi = ValueIteration(mdp)
    demo_vi.run_vi()
    demo_policy = demo_vi.policy
    demo_agent = FixedPolicyAgent(demo_policy, name="$\\pi_d$")

    # Make abstract agent.
    lambda_abstr_policy = get_lambda_policy(abstr_policy)
    prob_s_phi = ProbStateAbstraction(coding_distr)
    crisp_s_phi = convert_prob_sa_to_sa(prob_s_phi)
    abstr_agent = AbstractionWrapper(FixedPolicyAgent, state_abstr=crisp_s_phi, agent_params={"policy":lambda_abstr_policy, "name":"$\\pi_\\phi$"}, name_ext="")
    
    # Random agent.
    rand_agent = RandomAgent(mdp.get_actions())

    # Run.
    run_agents_on_mdp([demo_agent, abstr_agent, rand_agent], mdp, episodes=1, steps=1000)

    # Print state space sizes.
    print "\nState Spaces Sizes:"
    print "\t|S| =", len(demo_vi.get_states())
    print "\t|S_\\phi| =", get_sa_size_from_coding_distr(coding_distr)
    print "\t|S_\\phi|_crisp =", crisp_s_phi.get_num_abstr_states()
    print

def main():

    exp_type = "beta_plot_value"

    if exp_type == "beta_plot_state_size":
        # Makes a plot comparing beta (x-axis) vs. the abstract state space size (y-axis).
        make_barley_plot(_barley_s_phi_size_plot_wrapper, file_name="barley_beta_vs_size", y_label="$|S_\\phi|$")
    elif exp_type == "beta_plot_value":
        # Makes a plot comparing beta (x-axis) vs. the value of the abstract policy + state abstraction (y-axis).
        make_barley_plot(_barley_val_plot_wrapper, file_name="barley_beta_vs_value", y_label="$V(s_0)$")
    elif exp_type == "compare_policies":
        barley_compare_policies()

if __name__ == "__main__":
    main()
