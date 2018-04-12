# Python imports.
import math
from collections import defaultdict

from func_plotting import PlotFunc, plot_funcs

# ---------------------
# -- Print Functions --
# ---------------------

def print_pmf(pmf):
	for x in pmf.keys():
		print "x p(x):", x, pmf[x]

def print_coding_distr(coding_distribution):
	print "\n-- Coding Distr --"
	for x in coding_distribution.keys():
		print "x:", x
		for x_tilde in coding_distribution[x]:
			print "  code:", x_tilde
			print "  p(code | x):", coding_distribution[x][x_tilde]
			print
	print "--------------\n"

# ---------------------
# -- Misc. Functions --
# ---------------------

def distance(x_1, x_2):
	total_dist = 0.5
	if len(x_1) > len(x_2):
		for i, char in enumerate(x_1[1:]):
			if x_2[i-1] != char:
				total_dist += 1
	else:
		for i, char in enumerate(x_2[1:]):
			if x_1[i-1] != char:
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
	
	new_p_x["11"] = 0.6
	new_p_x["10"] = 0.05
	new_p_x["01"] = 0.05
	new_p_x["00"] = 0.3

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
	code_word = param_dict["code"]

	# Make message alphabet.
	all_x = ["00","01","10","11"]

	# Make coding alphabet.
	all_x_tilde = ["0","1"]

	# Init pmfs.
	p_of_x = init_p_of_x(all_x)
	p_of_x_tilde = init_p_of_x_tilde(all_x_tilde)
	coding_distribution = init_coding_distribution(all_x, all_x_tilde)

	# Blahut.
	for i in range(iters):
		# print "Round", i
		p_of_x_tilde = compute_prob_of_codes(p_of_x, coding_distribution, beta=beta)
		coding_distribution = compute_coding_distribution(p_of_x, p_of_x_tilde, coding_distribution, beta=beta)


	return p_of_x_tilde[code_word]


def main():


	# print "\n~~~ DONE ~~~"
	# print_pmf(p_of_x_tilde)
	# print
	# print_coding_distr(coding_distribution)

	# plot
	pf_zero = PlotFunc(blahut_arimoto, param_dict={"iters":100, "code":"0"}, x_min=0.0, x_max=5.0, x_interval=0.2, series_name="Pr(code = 0)")
	pf_one = PlotFunc(blahut_arimoto, param_dict={"iters":100, "code":"1"}, x_min=0.0, x_max=5.0, x_interval=0.2, series_name="Pr(code = 1)")

	plot_funcs([pf_zero, pf_one], title="$\\beta\\  vs. Pr(code)$", x_label="$\\beta$")

if __name__ == "__main__":
	main()