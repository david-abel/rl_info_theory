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
from simple_rl.agents import FixedPolicyAgent, RandomAgent
from simple_rl.tasks import FourRoomMDP, GridWorldMDP
from simple_rl.mdp import State
from simple_rl.planning import ValueIteration
from simple_rl.run_experiments import run_agents_on_mdp, evaluate_agent
from blahut_arimoto import print_coding_distr, print_pmf, mutual_info
from plot_barley import make_barley_val_plot
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


# ------------------
# -- Misc Helpers --
# ------------------

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

def run_barley(mdp, iters=100, beta=20.0, convergence_threshold=0.01):
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

# ---------------------
# -- Main Experiment --
# ---------------------

def barley_compare_policies():
    # Make MDP.
    mdp = FourRoomMDP(width=5, height=5, init_loc=(1, 1), goal_locs=[(5, 5)], gamma=0.9)

    # Run BARLEY.
    pmf_s_phi, coding_distr, abstr_policy = run_barley(mdp, iters=100, beta=3, convergence_threshold=0.00001)

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

def barley_visualize_abstr(beta=2.0):
    '''
    Args:
        beta (float)

    Summary:
        Visualizes the state abstraction found by barley.
    '''
    # Make MDP.
    mdp = FourRoomMDP(width=11, height=11, init_loc=(1, 1), goal_locs=[(11, 11)], gamma=0.9)

    # Run BARLEY.
    pmf_s_phi, coding_distr, abstr_policy = run_barley(mdp, iters=100, beta=beta, convergence_threshold=0.00001)
    lambda_abstr_policy = get_lambda_policy(abstr_policy)
    prob_s_phi = ProbStateAbstraction(coding_distr)
    crisp_s_phi = convert_prob_sa_to_sa(prob_s_phi)

    from simple_rl.abstraction.state_abs.sa_helpers import visualize_state_abstr_grid
    visualize_state_abstr_grid(mdp, crisp_s_phi)

def main():

    exp_type = "visualize_barley_abstr"

    if exp_type == "plot_barley_val":
        make_barley_val_plot()
    elif exp_type == "compare_policies":
        barley_compare_policies()
    elif exp_type == "visualize_barley_abstr":
        barley_visualize_abstr(beta=2.0)

if __name__ == "__main__":
    main()
