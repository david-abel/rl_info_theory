# RL and Info Theory
Code for running experiments associated with our 2019 AAAI paper: [State Abstraction as Compression in Apprenticeship Learning](https://david-abel.github.io/papers/rlit_aaai_2019.pdf), by David Abel, Dilip Arumugam, Kavosh Asadi, Yuu Jinnai, Michael L. Littman, and Lawson L.S. Wong.

To run experiments, you will need [simple_rl](https://github.com/david-abel/simple_rl), and to remake visuals, you will need [pygame](https://www.pygame.org/news) (both can be installed with pip).

To reproduce Figure 3a and 3b, run _info_sa.py_ with:
'''
exp_type = "plot_info_sa_val_and_num_states"
'''
in the main function. Then run _plot_info_sa.py_

To remake abstractions like the ones visualized in Figure 4b, 4c, and 4d, run _info_sa.py_ with:
'''
exp_type = "visualize_info_sa_abstr"
'''

To reproduce Figure 5a, run _deep/run_db.py_

Please contact Dave with any questions (david_abel@brown.edu).