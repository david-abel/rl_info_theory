'''
blahut_arimoto.py

Author: David Abel (github.com/david-abel)

An implementation of the Blahut-Arimoto algorithm from:
	Blahut, Richard (1972), "Computation of channel capacity and
	rate-distortion functions", IEEE Transactions on Information Theory, 18.
'''

# Python imports.
import math
import itertools
from collections import defaultdict


# ---------------------
# -- Print Functions --
# ---------------------

def print_pmf(pmf):
	print "\n-- p(x) --"
	for x in pmf.keys():
		print "  p(x = " + x + ") =", round(pmf[x], 3)
	print "----------\n"

def print_coding_distr(coding_distribution):
	print "\n-- Coding Distr --"
	for x in coding_distribution.keys():
		for x_tilde in coding_distribution[x]:
			print "  p(code = " + x_tilde + " | x = " + x + "):", round(coding_distribution[x][x_tilde], 3)
			print
	print "--------------"

# ---------------------
# -- Misc. Functions --
# ---------------------

def distance(x_1, x_2):
	'''
	Args:
		x_1 (str)
		x_2 (str)

	Returns:
		(float)
	'''
	total_dist = 0.5
	if len(x_1) > len(x_2):
		for i, char in enumerate(x_1[1:]):
			if x_2[i] != char:
				total_dist += 1
	else:
		for i, char in enumerate(x_2[1:]):
			if x_1[i] != char:
				total_dist += 1

	return total_dist

def _compute_denominator(x, p_of_x_tilde, beta):
	'''
	Args:
		x (str)
		p_of_x_tilde (dict)
		beta (float)

	Returns:
		(float)
	'''
	return sum([p_of_x_tilde[x_hat] * math.exp(- beta * distance(x, x_hat)) for x_hat in p_of_x_tilde.keys()])

def init_p_of_x(all_x):
	'''
	Args:
		all_x (list)

	Returns:
		(dict)
	'''

	new_p_x = defaultdict(float)
	
	new_p_x["11"] = 0.25
	new_p_x["10"] = 0.25
	new_p_x["01"] = 0.25
	new_p_x["00"] = 0.25

	# for x in all_x:
	# 	new_p_x[x] = float(x.count("1")) / len(all_x)

	return new_p_x

def init_p_of_x_tilde(all_x_tilde):
	'''
	Args:
		all_x_tilde (list)

	Returns:
		(dict)
	'''
	new_p_of_x_tilde = defaultdict(float)
	for x_tilde in all_x_tilde:
		new_p_of_x_tilde[x_tilde] = 1.0 / len(all_x_tilde)

	return new_p_of_x_tilde

def init_coding_distribution(all_x, all_x_tilde):
	'''
	Args:
		all_x (list)
		all_x_tilde (list)

	Returns:
		(float)
	'''
	new_coding_distribution = defaultdict(lambda : defaultdict(float))

	for x in all_x:
		for x_tilde in all_x_tilde:
			new_coding_distribution[x][x_tilde] = 1.0 / len(all_x_tilde)

	return new_coding_distribution

# -------------------------------
# -- Blahut Arimoto Main Steps --
# -------------------------------

def compute_prob_of_codes(p_of_x, coding_distribution, beta=.01):
	'''
	Args:
		p_of_x (dict):
		coding_distribution (dict)
		beta (float)

	Returns:
		(dict): pmf of x_tilde, p(\tilde{x}).
	'''
	new_p_of_x_tilde = defaultdict(float)
	for x_tilde in coding_distribution.values()[0].keys():
		new_p_of_x_tilde[x_tilde] = sum([p_of_x[x] * coding_distribution[x][x_tilde] for x in p_of_x.keys()])

	return new_p_of_x_tilde


def compute_coding_distribution(p_of_x, p_of_x_tilde, coding_distribution, beta=.01):
	'''
	Args:
		p_of_x (dict)
		p_of_x_tilde (dict)
		coding_distribution (dict)
		beta (float)

	Returns:
		(dict): new coding distribution.
	'''
	# Key: x
		# Value: dict, where:
			# Key: x_tilde
			# Value: probability
	new_coding_distribution = defaultdict(lambda : defaultdict(float))

	for x in p_of_x.keys():
		for x_tilde in p_of_x_tilde.keys():

			numerator = p_of_x_tilde[x_tilde] * math.exp(-beta * distance(x, x_tilde))
			denominator = _compute_denominator(x, p_of_x_tilde, beta)

			new_coding_distribution[x][x_tilde] = float(numerator) / denominator

	return new_coding_distribution


def blahut_arimoto(x, param_dict):
	'''
	Args:
		x (float)
		param_dict (dict)


	Returns:
		(float): A single value for p(code), for the first codeword.
	'''
	beta = x
	iters = param_dict["iters"]

	code_word = None
	if "code" in param_dict.keys():
		code_word = param_dict["code"]

	message = None
	if "message" in param_dict.keys():
		message = param_dict["message"]

	# Make message alphabet: length 2 bit sequences.
	all_x = ["".join(seq) for seq in itertools.product("01", repeat=2)]

	# Make coding alphabet: length 3 bit sequences.
	all_x_tilde = ["".join(seq) for seq in itertools.product("01", repeat=3)]

	# Init pmfs.
	p_of_x = init_p_of_x(all_x)
	p_of_x_tilde = init_p_of_x_tilde(all_x_tilde)
	coding_distribution = init_coding_distribution(all_x, all_x_tilde)

	# Blahut.
	for i in range(iters):
		# print "Round", i
		p_of_x_tilde = compute_prob_of_codes(p_of_x, coding_distribution, beta=beta)
		coding_distribution = compute_coding_distribution(p_of_x, p_of_x_tilde, coding_distribution, beta=beta)

	if code_word is None:
		# Return the two distributions from BA.
		return p_of_x_tilde, coding_distribution
	elif message is None:
		# Return p(code) for plotting.
		return p_of_x_tilde[code_word]
	else:
		# Return p(code | message) for plotting.
		return coding_distribution[message][code_word]

def make_ba_plot_func(code, message, iters=50):
	from func_plotting import PlotFunc

	param_dict = {"iters":iters, "code":str(code), "message":str(message)}
	plot_func_obj = PlotFunc(blahut_arimoto, param_dict=param_dict, x_min=0.0, x_max=5.0, x_interval=0.05, series_name="$Pr(code = " + str(code) + " \\mid " + str(message) + ")$")

	return plot_func_obj

def make_ba_plots():

	# Choose code-message combos to plot.
	funcs_to_plot = []
	iters = 10
	for message in ["".join(seq) for seq in itertools.product("01", repeat=2)][:2]:
		for code in ["".join(seq) for seq in itertools.product("01", repeat=3)][:4]:
			next_func = make_ba_plot_func(code, message, iters)
			funcs_to_plot.append(next_func)

	# Plot
	from func_plotting import plot_funcs
	plot_funcs(funcs_to_plot, title="$\\beta\\  vs. Pr$", x_label="$\\beta$")

def main():

	# Make plots
	plotting = True
	if plotting:
		make_ba_plots()
	else:
		beta = 0.0
		p_of_x_tilde, coding_distribution = blahut_arimoto(x=beta, param_dict={"iters":50})
		print_pmf(p_of_x_tilde)
		print_coding_distr(coding_distribution)
		print "beta =", beta
	
if __name__ == "__main__":
	main()