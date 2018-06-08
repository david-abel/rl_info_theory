# Python imports.
import os
import random

# Other imports.
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from simple_rl.abstraction.state_abs.ProbStateAbstractionClass import ProbStateAbstraction, convert_prob_sa_to_sa
from simple_rl.tasks import FourRoomMDP, GridWorldMDP
from simple_rl.planning import ValueIteration
from simple_rl.agents import FixedPolicyAgent
from simple_rl.run_experiments import evaluate_agent
from simple_rl.utils import chart_utils
from rlit_utils import write_datum_to_file, end_of_instance

# ---------------------
# -- BARLEY Plotting --
# ---------------------

def make_barley_val_plot(instances=3):
    '''
    Args:
        instances (int)

    Summary:
        Main plotting function for barley experiments.
    '''

    # Clear old results.
    for policy in ["crisp_val", "stochastic_val", "demo_val"]:
        if os.path.exists(os.path.join("barley_results/", str(policy)) + ".csv"):
            os.remove(os.path.join("barley_results/", str(policy)) + ".csv")

    # Set relevant params.
    grid_dim = 5
    mdp = FourRoomMDP(width=grid_dim, height=grid_dim, init_loc=(1, 1), goal_locs=[(grid_dim, grid_dim)], gamma=0.9)
    param_dict = {"mdp":mdp, "iters":200, "convergence_threshold":0.0001}

    # Choose beta interval.
    x_min, x_max, x_increment = 0.0, grid_dim*2, grid_dim / 10.0

    # Get and record demo policy value.
    vi = ValueIteration(mdp)
    vi.run_vi()
    demo_agent = FixedPolicyAgent(vi.policy)
    demo_val = evaluate_agent(demo_agent, mdp, instances=10)
    for beta in chart_utils.drange(x_min, x_max, x_increment):
        write_datum_to_file(file_name="demo_val", datum=demo_val, extra_dir="barley_results")

    # Compute barley.
    for instance in range(instances):

        random.jumpahead(1)
        # For each beta.
        for beta in chart_utils.drange(x_min, x_max, x_increment):
            crisp_value = _barley_val_plot_wrapper(x=beta, param_dict=dict(param_dict.items() + {"use_crisp_sa":False}.items()))
            stochastic_value = _barley_val_plot_wrapper(x=beta, param_dict=dict(param_dict.items() + {"use_crisp_sa":True}.items()))

            write_datum_to_file(file_name="crisp_val", datum=crisp_value, extra_dir="barley_results")
            write_datum_to_file(file_name="stochastic_val", datum=crisp_value, extra_dir="barley_results")

        end_of_instance("crisp_val", extra_dir="barley_results")
        end_of_instance("stochastic_val", extra_dir="barley_results")

    regular_series_name = "$\\pi^{\\phi-\\Pr}$"
    crispy_series_name = "$\\pi^{\\phi}$"

    # Plot.
    chart_utils.CUSTOM_TITLE = "Barley: beta vs. Value"
    chart_utils.X_AXIS_LABEL = "$\\beta$"
    chart_utils.Y_AXIS_LABEL = "$V(s_0)$"
    chart_utils.X_AXIS_START_VAL = x_min
    chart_utils.X_AXIS_INCREMENT = x_increment
    chart_utils.Y_AXIS_END_VAL = None

    chart_utils.make_plots("barley_results/", ["crisp_val", "stochastic_val", "demo_val"], cumulative=False, episodic=False, track_disc_reward=False)

def make_barley_plot_pf(plotting_helper_func, file_name="barley", x_label="$\\beta$", y_label="$|S_\\phi|$"):
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

    from barley import run_barley
    pmf_code, coding_distr, abstr_policy = run_barley(mdp, beta=x, iters=iters, convergence_threshold=convergence_threshold)

    # Make abstract agent.
    from barley import get_lambda_policy
    lambda_abstr_policy = get_lambda_policy(abstr_policy)
    prob_s_phi = ProbStateAbstraction(coding_distr)

    # Convert to crisp SA if needed.
    phi = convert_prob_sa_to_sa(prob_s_phi) if use_crisp_sa else prob_s_phi

    abstr_agent = AbstractionWrapper(FixedPolicyAgent, state_abstr=phi, agent_params={"policy":lambda_abstr_policy, "name":"$\\pi_\\phi$"}, name_ext="")
    
    # Compute value of abstract policy w/ coding distribution.
    value = evaluate_agent(agent=abstr_agent, mdp=mdp, instances=1000)

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
