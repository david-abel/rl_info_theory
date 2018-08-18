#!/usr/bin/env python

# Python imports.
import sys
import math
import numpy as np
import random
import os
import time
from collections import defaultdict

# Other imports.
from simple_rl.abstraction.state_abs.ProbStateAbstractionClass import ProbStateAbstraction, convert_prob_sa_to_sa
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from simple_rl.abstraction.AbstractValueIterationClass import AbstractValueIteration
from simple_rl.agents import FixedPolicyAgent, RandomAgent, QLearningAgent
from simple_rl.tasks import FourRoomMDP, GridWorldMDP, RandomMDP, TrenchOOMDP
from simple_rl.mdp import State
from simple_rl.planning import ValueIteration
from simple_rl.run_experiments import run_agents_on_mdp, evaluate_agent
from simple_rl.utils import chart_utils
from blahut_arimoto import print_coding_pmf, print_pmf, mutual_info
from plot_info_sa import make_info_sa_val_and_size_plots
from rlit_utils import *

distance_func = kl


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


def make_policy_det_max_policy(policy):
    '''
    Args:
        policy (dict)

    Returns:
        (dict)
    '''

    if not isinstance(policy, dict):
        policy = get_pmf_policy(policy)

    new_policy = {}
    for s in policy.keys():
        new_policy[s] = {policy[s].keys()[policy[s].values().index(max(policy[s].values()))]:1.0}

    return new_policy


# -----------------------------
# -- Iterative BA Like Steps --
# -----------------------------

def compute_prob_of_s_phi(pmf_s, phi_pmf, ground_states, abstract_states):
    '''
    Args:
        pmf_s (dict)
        phi_pmf (dict)
        ground_states (list)
        abstract_states (list)

    Returns:
        (dict)
    '''
    new_pmf_s_phi = defaultdict(float)

    for s_phi in abstract_states:
        new_pmf_s_phi[s_phi] = sum([pmf_s[s] * phi_pmf[s][s_phi] for s in ground_states])

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


def compute_phi_pmf(pmf_s, pmf_s_phi, demo_policy, abstr_policy, ground_states, abstract_states, beta, deterministic=False):
    '''
    Args:
        pmf_s (dict)
        pmf_s_phi (dict)
        demo_policy (dict)
        abstr_policy (dict)
        abstract_states (list)
        ground_states (list)
        beta (float)
        deterministic (bool): If true, run DIB, otherwise IB.

    Returns:
        (dict)

    '''
    new_phi_pmf = defaultdict(lambda: defaultdict(float))

    if deterministic:
        # Update according to DIB.
        for s in ground_states:
            # For each ground state, compute the best abstract state match.
            max_objective = float("-inf")
            best_s_phi = [s_phi for s_phi in abstract_states if pmf_s_phi[s_phi] > 0][0]

            for s_phi in abstract_states:
                if pmf_s_phi[s_phi] > 0:
                    # Compute the objective
                    next_obj = math.log(pmf_s_phi[s_phi]) - beta * distance_func(demo_policy[s], abstr_policy[s_phi])
                    if next_obj > max_objective:
                        max_objective = next_obj
                        best_s_phi = s_phi
    
            # Choose the abstract state that maximizes the objective.
            new_phi_pmf[s][best_s_phi] = 1

        return new_phi_pmf

    # Regular IB update.
    for s in ground_states:
        for s_phi in abstract_states:
            numerator = pmf_s_phi[s_phi] * math.exp(-beta * distance_func(demo_policy[s], abstr_policy[s_phi]))
            denominator = _compute_denominator(s, pmf_s_phi, demo_policy, abstr_policy, beta)

            if denominator == 0:
                print "Warning: division by zero in compute_phi_pmf."
                new_phi_pmf[s][s_phi] = 0
            else:    
                new_phi_pmf[s][s_phi] = numerator / denominator

    return new_phi_pmf


def compute_abstr_policy(demo_policy, ground_states, abstract_states, actions, phi_pmf, pmf_s, pmf_s_phi, deterministic=False):
    '''
    Args:
        demo_policy (lambda : s --> a)
        ground_states (list)
        abstract_states (list)
        actions (list)
        phi_pmf (dict)
        deterministic (bool): If true run DIB, else IB.

    Returns:
        (dict)
    '''
    abstr_policy = defaultdict(lambda: defaultdict(float))

    for s_phi in abstract_states:

        det_action_probs = defaultdict(float)
        for a in actions:

            total = 0.0
            for s in ground_states:
                if deterministic and phi_pmf[s][s_phi] > 0:
                    total += demo_policy[s][a] * pmf_s[s]
                else:
                    if pmf_s_phi[s_phi] == 0:
                        # Avoid division by zero.
                        continue
                    total += demo_policy[s][a] * phi_pmf[s][s_phi] * (pmf_s[s] / pmf_s_phi[s_phi])
            
            if deterministic:
                det_action_probs[a] = total
            else:
                abstr_policy[s_phi][a] = total

        if deterministic:
            # Grab the total mass of taking all actions.
            total_mass = sum(det_action_probs.values())
                    
            for a in actions:
                if total == 0:
                    # Probability of reaching s_phi is 0, so we don't care about the policy.
                    abstr_policy[s_phi][a] = 1.0 / len(actions)
                else:    
                    abstr_policy[s_phi][a] = det_action_probs[a] / total_mass

    return abstr_policy


# ------------------------
# -- Init Distributions --
# ------------------------

def get_stationary_rho_from_abstr_policy(abstr_policy_pmf, phi_pmf, mdp, ground_states, sample_rate=20, max_steps=40):
    '''
    Args:
        abstr_policy_pmf (dict): K=State ---> V=Pr(Action)
        phi_pmf (dict): K=State --> V=dict(K=State --> V=Pr)
        mdp (simple_rl.MDP)
        ground_states (list)

    Returns:
        (dict): policy (see above)
    '''
    # Make it crispy.
    new_phi = {}
    for s_g in phi_pmf.keys():
        new_phi[s_g] = phi_pmf[s_g].keys()[phi_pmf[s_g].values().index(max(phi_pmf[s_g].values()))]

    def new_policy(state):
        # Sample an abstract state.
        # sampled_s_phi_index = np.random.multinomial(1, phi_pmf[state].values()).tolist().index(1)
        # abstr_state = phi_pmf[state].keys()[sampled_s_phi_index]
        abstr_state = new_phi[state]

        # Sample an action.
        sampled_a_index = np.random.multinomial(1, abstr_policy_pmf[abstr_state].values()).tolist().index(1)
        return abstr_policy_pmf[abstr_state].keys()[sampled_a_index]

    return get_stationary_rho_from_policy(new_policy, mdp, ground_states, sample_rate, max_steps)

def get_stationary_rho_from_policy(policy, mdp, ground_states, sample_rate=200, max_steps=30):
    '''
    Args:
        policy (dict): K=State ---> V=Pr(Action)
        mdp (simple_rl.MDP)
        ground_states (list)
        sample_rate (int)
        max_steps (30)

    Returns:
        (dict): policy (see above)
    '''
    mdp.reset()
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


def init_random_phi(ground_states, deterministic=False):
    '''
    Args:
        num_ground_states (int)
        deterministic (bool): If true, run the DIB instead of IB.

    Returns:
        (tuple):
            (dict): phi : S --> [S_phi --> Pr]
            (set): list of abstract State objects.
    '''
    new_phi_pmf = defaultdict(lambda : defaultdict(float))
    num_ground_states = len(ground_states)
    abstr_states = set([])

    # Initialize the identity distribution
    for i, s_g in enumerate(ground_states):

        if deterministic:
            # Deterministic IB.
            ground_states_copy = ground_states[:]
            random.shuffle(ground_states_copy)

            for s_phi in xrange(len(ground_states)):
                new_phi_pmf[s_g][State(s_phi)] = int(s_phi == ground_states_copy.index(s_g))
                abstr_states.add(State(s_phi))
            continue
        else:
            # Regular IB.
            new_multinomial_counts = np.random.multinomial(num_ground_states * 2, [1.0/num_ground_states]*num_ground_states, size=1)[0]
            s_phi_distr = [float(i) / num_ground_states * 2 for i in new_multinomial_counts]

        # Set probabilities for each s_g --> s_phi pair.
        for s_phi in xrange(len(ground_states)):
            new_phi_pmf[s_g][State(s_phi)] = s_phi_distr[s_phi]
            abstr_states.add(State(s_phi))
    
    return new_phi_pmf, list(abstr_states)


def init_identity_phi(ground_states):
    '''
    Args:
        num_ground_states (int)

    Returns:
        (float)
    '''
    new_phi_pmf = defaultdict(lambda : defaultdict(float))

    # Initialize the identity distribution
    for i, s_g in enumerate(ground_states):
        for s_phi in xrange(len(ground_states)):
            new_phi_pmf[s_g][State(s_phi)] =  int(i == s_phi)

    return new_phi_pmf


def init_uniform_phi(num_ground_states):
    '''
    Args:
        num_ground_states (int)

    Returns:
        (float)
    '''
    new_phi_pmf = defaultdict(lambda : defaultdict(float))

    # Initialize the identity distribution
    for s_g in range(num_ground_states):
        for s_phi in range(num_ground_states):
            new_phi_pmf[s_g][State(s_phi)] =  1.0 / num_ground_states

    return new_phi_pmf

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


# ------------------------
# -- info_sa Main Steps --
# ------------------------

def run_info_sa(mdp, demo_policy_lambda, deterministic=False, iters=500, beta=20.0, convergence_threshold=0.01, deterministic_ib=False):
    '''
    Args:
        mdp (simple_rl.MDP)
        demo_policy (lambda : simple_rl.State --> str)
        deterministic (bool): If deterministic, we run the Deterministic Info Bottlekneck instead of regular IB.
        iters (int)
        beta (float)
        convergence_threshold (float): When all three distributions satisfy
            L1(p_{t+1}, p_t) < @convergence_threshold, we stop iterating.
        deterministic_ib (bool): If true, run DIB, else IB.
        is_agent_in_control (bool): If True, the agent's actions dictate the source distribution on states. Otherwise,
            we assume the demonstrator policy controls the source distribution on states (which is then fixed).

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

    # Get state and action space.
    vi = ValueIteration(mdp)
    ground_states = vi.get_states()
    num_ground_states = len(ground_states)
    actions = mdp.get_actions()

    # Get pmf demo policy and stationary distribution.
    demo_policy_pmf = get_pmf_policy(demo_policy_lambda, ground_states, actions)
    pmf_s = get_stationary_rho_from_policy(demo_policy_lambda, mdp, ground_states)

    # Init distributions.
    phi_pmf, abstract_states = init_random_phi(ground_states, deterministic=deterministic_ib)
    pmf_s_phi = compute_prob_of_s_phi(pmf_s, phi_pmf, ground_states, abstract_states)
    abstr_policy_pmf = compute_abstr_policy(demo_policy_pmf, ground_states, abstract_states, actions, phi_pmf, pmf_s, pmf_s_phi, deterministic=deterministic_ib)

    # info_sa.
    for i in range(iters):
        print 'Iteration {0} of {1}'.format(i+1, iters)

        # (A) Compute \phi.
        next_phi_pmf = compute_phi_pmf(pmf_s, pmf_s_phi, demo_policy_pmf, abstr_policy_pmf, ground_states, abstract_states, beta=beta, deterministic=deterministic_ib)

        # (B) Compute \rho(s).
        next_pmf_s_phi = compute_prob_of_s_phi(pmf_s, next_phi_pmf, ground_states, abstract_states)

        # (C) Compute \pi_\phi.
        next_abstr_policy_pmf = compute_abstr_policy(demo_policy_pmf, ground_states, abstract_states, actions, next_phi_pmf, pmf_s, next_pmf_s_phi, deterministic=deterministic_ib)

        # Convergence checks.
        coding_update_delta = max([l1_distance(next_phi_pmf[s], phi_pmf[s]) for s in ground_states])
        policy_update_delta = max([l1_distance(next_abstr_policy_pmf[s_phi], abstr_policy_pmf[s_phi]) for s_phi in abstract_states])
        state_distr_update_delta = l1_distance(next_pmf_s_phi, pmf_s_phi)

        # Debugging.
        is_coding_converged = coding_update_delta < convergence_threshold
        is_policy_converged = policy_update_delta < convergence_threshold
        is_pmf_s_phi_converged = state_distr_update_delta < convergence_threshold

        # Update pointers.
        phi_pmf = next_phi_pmf
        pmf_s_phi = next_pmf_s_phi
        abstr_policy_pmf = next_abstr_policy_pmf

        if is_coding_converged and is_policy_converged and is_pmf_s_phi_converged:
            print "\tinfo_sa Converged."
            break

    return pmf_s_phi, phi_pmf, abstr_policy_pmf


# ---------------------
# -- Main Experiment --
# ---------------------

def info_sa_compare_policies(mdp, demo_policy_lambda, beta=3.0, deterministic_ib=False):
    '''
    Args:
        mdp (simple_rl.MDP)
        demo_policy_lambda (lambda : simple_rl.State --> str)
        beta (float)
        deterministic_ib (bool): If True, run DIB, else IB.

    Summary:
        Runs info_sa and compares the value of the found policy with the demonstrator policy.
    '''
    # Run info_sa.
    pmf_s_phi, phi_pmf, abstr_policy_pmf = run_info_sa(mdp, demo_policy_lambda, iters=500, beta=beta, convergence_threshold=0.00001, deterministic_ib=deterministic_ib)

    # Make demonstrator agent and random agent.
    demo_agent = FixedPolicyAgent(demo_policy_lambda, name="$\\pi_d$")
    rand_agent = RandomAgent(mdp.get_actions(), name="$\\pi_u$")

    # Make abstract agent.
    lambda_abstr_policy = get_lambda_policy(abstr_policy_pmf)
    prob_s_phi = ProbStateAbstraction(phi_pmf)
    crisp_s_phi = convert_prob_sa_to_sa(prob_s_phi)
    abstr_agent = AbstractionWrapper(FixedPolicyAgent, state_abstr=crisp_s_phi, agent_params={"policy":lambda_abstr_policy, "name":"$\\pi_\\phi$"}, name_ext="")
    
    # Run.
    run_agents_on_mdp([demo_agent, abstr_agent, rand_agent], mdp, episodes=1, steps=1000)


    non_zero_abstr_states = [x for x in pmf_s_phi.values() if x > 0]
    # Print state space sizes.
    demo_vi = ValueIteration(mdp)
    print "\nState Spaces Sizes:"
    print "\t|S| =", demo_vi.get_num_states()
    print "\tH(S_\\phi) =", entropy(pmf_s_phi)
    print "\t|S_\\phi|_crisp =", crisp_s_phi.get_num_abstr_states()
    print "\tdelta_min =", min(non_zero_abstr_states)
    print "\tnum non zero states =", len(non_zero_abstr_states)
    print

def info_sa_visualize_abstr(mdp, demo_policy, beta=2.0, deterministic_ib=False):
    '''
    Args:
        mdp (simple_rl.MDP)
        demo_policy (lambda : simple_rl.State --> str)
        beta (float)
        deterministic_ib (bool)

    Summary:
        Visualizes the state abstraction found by info_sa using pygame.
    '''
    # Run info_sa.
    pmf_s_phi, phi_pmf, abstr_policy = run_info_sa(mdp, demo_policy, iters=500, beta=beta, convergence_threshold=0.00001, deterministic_ib=deterministic_ib)
    lambda_abstr_policy = get_lambda_policy(abstr_policy)
    prob_s_phi = ProbStateAbstraction(phi_pmf)
    crisp_s_phi = convert_prob_sa_to_sa(prob_s_phi)

    from simple_rl.abstraction.state_abs.sa_helpers import visualize_state_abstr_grid
    visualize_state_abstr_grid(mdp, crisp_s_phi)

def info_sa_planning_experiment(min_grid_size=5, max_grid_size=11, beta=10.0):
    '''
    Args:
        min_grid_size (int)
        max_grid_size (int)
        beta (float): Hyperparameter for InfoSA.

    Summary:
        Writes num iterations and time (seconds) for planning with and without abstractions.
    '''
    vanilla_file = "vi.csv"
    sa_file = "vi-$\\phi$.csv"
    file_prefix = os.path.join("results", "planning-four_room")
    
    clear_files(dir_name=file_prefix)

    for grid_dim in xrange(min_grid_size, max_grid_size + 1):
        # ======================
        # == Make Environment ==
        # ======================
        mdp = FourRoomMDP(width=grid_dim, height=grid_dim, init_loc=(1, 1), goal_locs=[(grid_dim, grid_dim)], gamma=0.9)
        
        # Get demo policy.
        vi = ValueIteration(mdp)
        vi.run_vi()
        demo_policy = get_lambda_policy(make_det_policy_eps_greedy(vi.policy, vi.get_states(), mdp.get_actions(), epsilon=0.2))

        # =======================
        # == Make Abstractions ==
        # =======================
        pmf_s_phi, phi_pmf, abstr_policy = run_info_sa(mdp, demo_policy, iters=500, beta=beta, convergence_threshold=0.00001)
        lambda_abstr_policy = get_lambda_policy(abstr_policy)
        prob_s_phi = ProbStateAbstraction(phi_pmf)
        crisp_s_phi = convert_prob_sa_to_sa(prob_s_phi)

        # ============
        # == Run VI ==
        # ============
        vanilla_vi = ValueIteration(mdp, delta=0.0001, sample_rate=25)
        sa_vi = AbstractValueIteration(ground_mdp=mdp, state_abstr=crisp_s_phi, delta=0.0001, vi_sample_rate=25, amdp_sample_rate=25)

        # ==========
        # == Plan ==
        # ==========
        print "Running VIs."
        start_time = time.clock()
        vanilla_iters, vanilla_val = vanilla_vi.run_vi()
        vanilla_time = round(time.clock() - start_time, 2)

        mdp.reset()
        start_time = time.clock()
        sa_iters, sa_abs_val = sa_vi.run_vi()
        sa_time = round(time.clock() - start_time, 2)
        sa_val = evaluate_agent(FixedPolicyAgent(sa_vi.policy), mdp, instances=25)

        print "\n" + "*"*20
        print "Vanilla", "\n\t Iters:", vanilla_iters, "\n\t Value:", round(vanilla_val, 4), "\n\t Time:", vanilla_time
        print 
        print "Phi:", "\n\t Iters:", sa_iters, "\n\t Value:", round(sa_val, 4), "\n\t Time:", sa_time
        print "*"*20 + "\n\n"

        write_datum(os.path.join(file_prefix, "iters", vanilla_file), vanilla_iters)
        write_datum(os.path.join(file_prefix, "iters", sa_file), sa_iters)

        write_datum(os.path.join(file_prefix, "times", vanilla_file), vanilla_time)
        write_datum(os.path.join(file_prefix, "times", sa_file), sa_time)

def learn_w_abstr(mdp, demo_policy, beta_list=[0.0, 2.0, 10.0], deterministic_ib=False):
    '''
    Args:
        mdp (simple_rl.MDP)
        demo_policy (lambda : simple_rl.State --> str)
        beta_list (list)
        deterministic_ib (bool)

    Summary:
        Computes a state abstraction for the given beta and compares Q-Learning with and without the abstraction.
    '''
    # Run info_sa.
    dict_of_phi_pmfs = {}
    for beta in beta_list:
        pmf_s_phi, phi_pmf, abstr_policy_pmf = run_info_sa(mdp, demo_policy, iters=300, beta=beta, convergence_threshold=0.0001, deterministic_ib=deterministic_ib)

        # Translate abstractions.
        prob_s_phi = ProbStateAbstraction(phi_pmf)
        crisp_s_phi = convert_prob_sa_to_sa(prob_s_phi)
        dict_of_phi_pmfs[beta] = crisp_s_phi

    # Make agents.
    demo_agent = FixedPolicyAgent(demo_policy, name="$\\pi_d$")
    ql_agent = QLearningAgent(mdp.get_actions())
    agent_dict = {}
    for beta in beta_list:
        beta_phi = dict_of_phi_pmfs[beta]
        ql_abstr_agent = AbstractionWrapper(QLearningAgent, state_abstr=dict_of_phi_pmfs[beta], agent_params={"actions":mdp.get_actions()}, name_ext="-$\\phi_{\\beta = " + str(beta) + "}$")
        agent_dict[beta] = ql_abstr_agent

    # Learn.
    run_agents_on_mdp([ql_agent] + agent_dict.values(), mdp, episodes=2000, steps=50, instances=50)

    # Print num abstract states.
    for beta in dict_of_phi_pmfs.keys():
        print "beta |S_phi|:", beta, dict_of_phi_pmfs[beta].get_num_ground_states()
    print

def main():

    # Make MDP.
    grid_dim = 11
    mdp = FourRoomMDP(width=grid_dim, height=grid_dim, init_loc=(1, 1), goal_locs=[(grid_dim, grid_dim)], gamma=0.9)

    # Experiment Type.
    exp_type = "plot_info_sa_val_and_num_states"

    # For comparing policies and visualizing.
    beta = 15.0
    deterministic_ib = True

    # For main plotting experiment.
    beta_range = list(chart_utils.drange(0.0, 5.0, 0.5))
    instances = 10

    # Get demo policy.
    vi = ValueIteration(mdp)
    _, val = vi.run_vi()
    demo_policy = get_lambda_policy(make_det_policy_eps_greedy(vi.policy, vi.get_states(), mdp.get_actions(), epsilon=0.2))

    if exp_type == "plot_info_sa_val_and_num_states":
        # Makes the main two plots.
        make_info_sa_val_and_size_plots(mdp, demo_policy, beta_range, instances=instances)
    elif exp_type == "compare_policies":
        # Makes a plot comparing value of pi-phi combo from info_sa with \pi_d.
        info_sa_compare_policies(mdp, demo_policy, beta=beta, deterministic_ib=deterministic_ib)
    elif exp_type == "visualize_info_sa_abstr":
        # Visualize the state abstraction found by info_sa.
        info_sa_visualize_abstr(mdp, demo_policy, beta=beta, deterministic_ib=deterministic_ib)
    elif exp_type == "learn_w_abstr":
        # Run learning experiments for different settings of \beta.
        learn_w_abstr(mdp, demo_policy, deterministic_ib=deterministic_ib)
    elif exp_type == "planning":
        info_sa_planning_experiment()

if __name__ == "__main__":
    main()
