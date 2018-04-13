'''
blahut_arimoto.py

Author: David Abel (github.com/david-abel)

An implementation of the Blahut-Arimoto algorithm from:
	Blahut, Richard (1972), "Computation of channel capacity and
	rate-distortion functions", IEEE Transactions on Information Theory, 18.
'''

# Python imports.
import math
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
	
	new_p_x["11"] = 0.5
	new_p_x["10"] = 0.3
	new_p_x["01"] = 0.1
	new_p_x["00"] = 0.1

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

	if code_word is None:
		# Return the two distributions from BA.
		return p_of_x_tilde, coding_distribution
	elif message is None:
		# Return p(code) for plotting.
		return p_of_x_tilde[code_word]
	else:
		# Return p(code | message) for plotting.
		return coding_distribution[message][code_word]

def make_ba_plots():
	from func_plotting import PlotFunc, plot_funcs

	param_dict_zero_oo = {"iters":50, "code":"0", "message":"11"}
	param_dict_one_oo = {"iters":50, "code":"1", "message":"11"}
	param_dict_zero_zo = {"iters":50, "code":"0", "message":"10"}
	param_dict_one_zo = {"iters":50, "code":"1", "message":"10"}
	param_dict_zero_oz = {"iters":50, "code":"0", "message":"01"}
	param_dict_one_oz = {"iters":50, "code":"1", "message":"01"}
	param_dict_zero_zz = {"iters":50, "code":"0", "message":"00"}
	param_dict_one_zz = {"iters":50, "code":"1", "message":"00"}

	title = "$\\beta\\  vs. Pr$"

	# plot p(code)
	pf_zero_oo = PlotFunc(blahut_arimoto, param_dict=param_dict_zero_oo, x_min=0.0, x_max=5.0, x_interval=0.1, series_name="$Pr(code = 0 \\mid 11)$")
	pf_one_oo = PlotFunc(blahut_arimoto, param_dict=param_dict_one_oo, x_min=0.0, x_max=5.0, x_interval=0.1, series_name="$Pr(code = 1 \\mid 11)$")
	pf_zero_zo = PlotFunc(blahut_arimoto, param_dict=param_dict_zero_zo, x_min=0.0, x_max=5.0, x_interval=0.1, series_name="$Pr(code = 0 \\mid 10)$")
	pf_one_zo = PlotFunc(blahut_arimoto, param_dict=param_dict_one_zo, x_min=0.0, x_max=5.0, x_interval=0.1, series_name="$Pr(code = 1 \\mid 10)$")
	# pf_zero_oz = PlotFunc(blahut_arimoto, param_dict=param_dict_zero_oz, x_min=0.0, x_max=5.0, x_interval=0.1, series_name="$Pr(code = 0 \\mid 01)$")
	# pf_one_oz = PlotFunc(blahut_arimoto, param_dict=param_dict_one_oz, x_min=0.0, x_max=5.0, x_interval=0.1, series_name="$Pr(code = 1 \\mid 01)$")
	# pf_zero_zz = PlotFunc(blahut_arimoto, param_dict=param_dict_zero_zz, x_min=0.0, x_max=5.0, x_interval=0.1, series_name="$Pr(code = 0 \\mid 00)$")
	# pf_one_zz = PlotFunc(blahut_arimoto, param_dict=param_dict_one_zz, x_min=0.0, x_max=5.0, x_interval=0.1, series_name="$Pr(code = 1 \\mid 00)$")

	plot_funcs([pf_zero_oo, pf_one_oo, pf_zero_zo, pf_one_zo], title=title, x_label="$\\beta$")

def main():

	# Make plots
	plotting = False
	if plotting:
		make_ba_plots()
	else:
		beta = 0.5
		p_of_x_tilde, coding_distribution = blahut_arimoto(x=beta, param_dict={"iters":50})
		print_pmf(p_of_x_tilde)
		print_coding_distr(coding_distribution)
		print "beta =", beta
	
if __name__ == "__main__":
	main()