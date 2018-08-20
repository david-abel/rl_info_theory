# Python imports.
from collections import defaultdict
import numpy as np

# Other imports.
import info_sa
from simple_rl.mdp import State
from simple_rl.tasks import FourRoomMDP
from simple_rl.planning import ValueIteration
from simple_rl.abstraction.state_abs.ProbStateAbstractionClass import ProbStateAbstraction, convert_prob_sa_to_sa
from rlit_utils import *

def get_stationary_rho_ground_states_from_abstr_policy(policy_pmf, mdp, ground_states, phi_pmf, sample_rate=200, max_steps=30):
    '''
    Args:
        policy_pmf (dict): K=State ---> V=Pr(Action)
        mdp (simple_rl.MDP)
        ground_states (list)
        phi_pmf (dict)
        sample_rate (int)
        max_steps (30)

    Returns:
        (dict): stationary distribution on S_g under policy_pmf.
    '''
    
    # Initialize.
    prob_s_phi = ProbStateAbstraction(phi_pmf)
    crisp_s_phi = convert_prob_sa_to_sa(prob_s_phi)
    mdp.reset()
    s = mdp.get_init_state()
    rho = defaultdict(float)
    policy_lambda = info_sa.get_lambda_policy(policy_pmf)

    total = 0
    for _ in range(sample_rate):
        num_steps = 0
        while not s.is_terminal() and num_steps < max_steps:
            rho[s] += 1.0
            total += 1
            s_abstr = crisp_s_phi.phi(s)
            _, s = mdp.execute_agent_action(policy_lambda(s_abstr))
            num_steps += 1

        if s.is_terminal():
            rho[s] += 1.0
            total += 1

        mdp.reset()
        s = mdp.get_init_state()

    for k in ground_states:
        rho[k] = rho[k] / total

    return rho

def run_agent_in_control_info_sa(mdp, demo_policy_lambda, rounds=10, iters=500, beta=20.0, round_convergence_thresh=0.1, iter_convergence_thresh=0.000001, is_deterministic_ib=False):
    '''
    Args:
        mdp (simple_rl.MDP)
        demo_policy (lambda : simple_rl.State --> str)
        rounds (int): mumber of times to run info_sa to convergence.
        iters (int): max number of iterations per round.
        beta (float)
        round_convergence_thresh (float): Determines when to stop rounds.
        iter_convergence_thresh (float): When all three distributions satisfy
            L1(p_{t+1}, p_t) < @convergence_threshold, we stop iterating.
        is_deterministic_ib (bool): If true, run DIB, else IB.
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
    demo_policy_pmf = info_sa.get_pmf_policy(demo_policy_lambda, ground_states, actions)

    # Init distributions.
    phi_pmf, abstract_states = info_sa.init_random_phi(ground_states, deterministic=is_deterministic_ib)
    pmf_s_phi = init_random_rho_phi(abstract_states)
    abstr_policy_pmf = init_random_pi(abstract_states, actions)

    # Get stationary distr of agent.
    pmf_s = get_stationary_rho_ground_states_from_abstr_policy(abstr_policy_pmf, mdp, ground_states, phi_pmf)

    # For each round.
    round_num = 0
    while round_num < rounds:
        print "\tRound", round_num + 1, "of", str(rounds) + "."

        # info_sa.
        for i in range(iters):
            print "\t\titer", i

            # (A) Compute \phi.
            next_phi_pmf = info_sa.compute_phi_pmf(pmf_s, pmf_s_phi, demo_policy_pmf, abstr_policy_pmf, ground_states, abstract_states, beta=beta, deterministic=is_deterministic_ib)

            # (B) Compute \rho(s).
            next_pmf_s_phi = info_sa.compute_prob_of_s_phi(pmf_s, next_phi_pmf, ground_states, abstract_states)

            # (C) Compute \pi_\phi.
            next_abstr_policy_pmf = info_sa.compute_abstr_policy(demo_policy_pmf, ground_states, abstract_states, actions, next_phi_pmf, pmf_s, next_pmf_s_phi, deterministic=is_deterministic_ib)

            # Convergence checks.
            coding_update_delta = max([l1_distance(next_phi_pmf[s], phi_pmf[s]) for s in ground_states])
            policy_update_delta = max([l1_distance(next_abstr_policy_pmf[s_phi], abstr_policy_pmf[s_phi]) for s_phi in abstract_states])
            state_distr_update_delta = l1_distance(next_pmf_s_phi, pmf_s_phi)

            # Debugging.
            is_coding_converged = coding_update_delta < iter_convergence_thresh
            is_policy_converged = policy_update_delta < iter_convergence_thresh
            is_pmf_s_phi_converged = state_distr_update_delta < iter_convergence_thresh

            # Update pointers.
            phi_pmf = next_phi_pmf
            pmf_s_phi = next_pmf_s_phi
            abstr_policy_pmf = next_abstr_policy_pmf

            if is_coding_converged and is_policy_converged and is_pmf_s_phi_converged:
                break

        round_num += 1
        next_pmf_s = get_stationary_rho_ground_states_from_abstr_policy(abstr_policy_pmf, mdp, ground_states, phi_pmf)

        print "\t\t rho gap:", round(l1_distance(next_pmf_s, pmf_s), 3)
        if l1_distance(next_pmf_s, pmf_s) < round_convergence_thresh:
            print "\t\tConverged."
            break

        pmf_s = next_pmf_s

    return pmf_s_phi, phi_pmf, abstr_policy_pmf


# ----------------------------
# -- Initializing Functions --
# ----------------------------

def init_random_rho_phi(states):
    '''
    Args:
        states (list)

    Returns:
        (dict)
    '''
    new_rho_phi = defaultdict(float)
    state_distr = np.random.dirichlet([0.5] * len(states)).tolist()

    for i in range(len(states)):
        new_rho_phi[State(i)] = state_distr[i]

    return new_rho_phi

def init_random_pi(states, actions):
    '''
    Args:
        pmf
        actions (list)

    Returns:
        (dict)
    '''
    new_pi = defaultdict(lambda: defaultdict(float))

    for s in states:
        action_distr = np.random.dirichlet([0.5] * len(actions)).tolist()
        for i, a in enumerate(actions):
            new_pi[s][a] = action_distr[i]

    return new_pi


# ----------
# -- Main --
# ----------

def main():
    # Make MDP.
    grid_dim = 11
    mdp = FourRoomMDP(width=grid_dim, height=grid_dim, init_loc=(1, 1), slip_prob=0.05, goal_locs=[(grid_dim, grid_dim)], gamma=0.99)

    # Get demo policy.
    vi = ValueIteration(mdp)
    _, val = vi.run_vi()

    # Epsilon greedy policy
    demo_policy = info_sa.get_lambda_policy(info_sa.make_det_policy_eps_greedy(vi.policy, vi.get_states(), mdp.get_actions(), epsilon=0.1))

    # Run.
    run_agent_in_control_info_sa(mdp, demo_policy, beta=5.0, is_deterministic_ib=True)

if __name__ == "__main__":
    main()
