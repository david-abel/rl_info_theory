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

message_len, code_len = 2, 1

# ---------------------
# -- Print Functions --
# ---------------------

def print_pmf(pmf):
	print "\n-- p(x) --"
	for x in pmf.keys():
		print "  p(x = " + x + ") =", round(pmf[x], 3)
	print "----------\n"

def print_coding_distr(coding_distr):
	print "\n-- Coding Distr --"
	for x in coding_distr.keys():
		for code in coding_distr[x]:
			print "  p(code = " + code + " | x = " + x + "):", round(coding_distr[x][code], 3)
			print
	print "--------------"

# -----------------------
# -- Info Theory Funcs --
# -----------------------

def entropy(pmf):
	'''
	Args:
		pmf (dict)

	Returns:
		(float)
	'''
	total = 0
	for x in pmf.keys():
		total -= pmf[x] * math.log(pmf[x],2)

	return total

def conditional_entropy(pmf_y_given_x, pmf_x):
	'''
	Args:
		pmf_y_given_x
		pmf_x

	Returns:
		(float)
	'''
	total = 0
	for x in pmf_x.keys():
		for y in pmf_y_given_x[x].keys():
			total -= pmf_y_given_x[x][y] * pmf_x[x] * math.log(pmf_y_given_x[x][y], 2)

	return total

def mutual_info(pmf_x, pmf_y, pmf_x_given_y):
	'''
	Args:
		pmf_x (dict)
		pmf_y (dict)

	Returns:
		(float)
	'''
	total = 0
	for x in pmf_x:
		for y in pmf_y:
			log_numer = pmf_x_given_y[y][x] * pmf_x[x]
			log_denom = pmf_y[y] * pmf_x[x]
			total += pmf_x_given_y[y][x] * pmf_x[x] * math.log(log_numer / log_denom, 2)

	return total

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

def init_pmf_x(all_x):
	'''
	Args:
		all_x (list)

	Returns:
		(dict)
	'''

	new_pmf_x = defaultdict(float)
	for x in all_x:
		new_pmf_x[x] = float(x.count("1")) / len(all_x)

	return new_pmf_x

def init_pmf_code(all_codes):
	'''
	Args:
		all_codes (list)

	Returns:
		(dict)
	'''
	new_pmf_code = defaultdict(float)
	for code in all_codes:
		new_pmf_code[code] = 1.0 / len(all_codes)

	return new_pmf_code

def init_coding_distr(all_x, all_codes):
	'''
	Args:
		all_x (list)
		all_codes (list)

	Returns:
		(float)
	'''
	new_coding_distr = defaultdict(lambda : defaultdict(float))

	for x in all_x:
		for code in all_codes:
			new_coding_distr[x][code] = 1.0 / len(all_codes)

	return new_coding_distr

# -------------------------------
# -- Blahut Arimoto Main Steps --
# -------------------------------

def _compute_denominator(x, pmf_code, beta):
	'''
	Args:
		x (str)
		pmf_code (dict)
		beta (float)

	Returns:
		(float)
	'''
	return sum([pmf_code[code_hat] * math.exp(- beta * distance(x, code_hat)) for code_hat in pmf_code.keys()])

def compute_prob_of_codes(pmf_x, coding_distr, beta=.01):
	'''
	Args:
		pmf_x (dict):
		coding_distr (dict)
		beta (float)

	Returns:
		(dict): pmf of code, p(code).
	'''
	new_pmf_code = defaultdict(float)
	for code in coding_distr.values()[0].keys():
		new_pmf_code[code] = sum([pmf_x[x] * coding_distr[x][code] for x in pmf_x.keys()])

	return new_pmf_code

def compute_coding_distr(pmf_x, pmf_code, coding_distr, beta=.01):
	'''
	Args:
		pmf_x (dict)
		pmf_code (dict)
		coding_distr (dict)
		beta (float)

	Returns:
		(dict): new coding distribution.
	'''
	# Key: x
		# Value: dict, where:
			# Key: code
			# Value: probability
	new_coding_distr = defaultdict(lambda : defaultdict(float))

	for x in pmf_x.keys():
		for code in pmf_code.keys():

			numerator = pmf_code[code] * math.exp(-beta * distance(x, code))
			denominator = _compute_denominator(x, pmf_code, beta)

			new_coding_distr[x][code] = float(numerator) / denominator

	return new_coding_distr

def blahut_arimoto(beta, iters=50):
	'''
	Args:
		beta (float)
		message_len (int)
		code_len (int)
		iters (int)

	Returns:
		(tuple):
			1 --> (dict) : Pr(code)
			2 --> (dict) : Pr(code | x)
	'''

	# Make message alphabet: length 2 bit sequences.
	all_x = ["".join(seq) for seq in itertools.product("01", repeat=message_len)]

	# Make coding alphabet: length 3 bit sequences.
	all_codes = ["".join(seq) for seq in itertools.product("01", repeat=code_len)]

	# Init pmfs.
	pmf_x = init_pmf_x(all_x)
	pmf_code = init_pmf_code(all_codes)
	coding_distr = init_coding_distr(all_x, all_codes)

	# Blahut.
	for i in range(iters):
		# print "Round", i
		pmf_code = compute_prob_of_codes(pmf_x, coding_distr, beta=beta)
		coding_distr = compute_coding_distr(pmf_x, pmf_code, coding_distr, beta=beta)

	# Return the two distributions from BA.
	return pmf_code, coding_distr

# --------------
# -- Plotting --
# --------------

def _blahut_arimoto_plot_func_wrapper(x, param_dict):
	'''
	Args:
		x (float): stands for $\beta$ in the BA algorithm.
		param_dict (dict): contains relevant parameters for plotting.

	Returns:
		(dict)

	Notes:
		This serves as a wrapper to cooperate with PlotFunc.
	'''

	pmf_code, coding_distr = blahut_arimoto(beta=x)

	if "message" not in param_dict.keys() and "code" in param_dict.keys():
		# Just plotting the code pmf.
		code = param_dict["code"]
		return pmf_code[code]

	elif "message" in param_dict.keys() and "code" in param_dict.keys():
		# Plot the coding pmf for the given code/message combo.
		message = param_dict["message"]
		code = param_dict["code"]

		return coding_distr[message][code]

def _entropy_of_ba_pmf_wrapper(x, param_dict):
	'''
	Args:
		x (float)
		param_dict (dict)

	Returns:
		(float): Retrieves entropy of the computed pmf, Pr(code)
	'''
	pmf_code, _ = blahut_arimoto(beta=x)
	return entropy(pmf_code)

def _entropy_of_ba_conditional_pmf_wrapper(x, param_dict):
	'''
	Args:
		x (float)
		param_dict (dict)

	Returns:
		(float): Retrieves conditional entropy of Pr(c | m)
	'''
	pmf_code, coding_distr = blahut_arimoto(beta=x)

	# Make message alphabet: length 2 bit sequences.
	all_x = ["".join(seq) for seq in itertools.product("01", repeat=message_len)]
	pmf_x = init_pmf_x(all_x)

	return conditional_entropy(coding_distr, pmf_x)

def make_ba_plot():
	'''
	Summary:
		Creates a plot showcasing the pmfs of the distributions
		computed by the blahut arimoto algorithm.
	'''
	from func_plotting import PlotFunc

	# Choose code-message combos to plot.
	funcs_to_plot = []
	for message in ["".join(seq) for seq in itertools.product("01", repeat=message_len)][:1]:
		for code in ["".join(seq) for seq in itertools.product("01", repeat=code_len)][:1]:
			
			# Set relevant params.
			param_dict = {"code":str(code), "message":str(message)}

			# Make plot func object.
			plot_ba = PlotFunc(_blahut_arimoto_plot_func_wrapper, param_dict=param_dict, x_min=0.0, x_max=5.0, x_interval=0.2, series_name="$Pr(c = " + str(code) + " \\mid m = " + str(message) + ")$")
			plot_entropy = PlotFunc(_entropy_of_ba_pmf_wrapper, param_dict=param_dict, x_min=0.0, x_max=5.0, x_interval=0.2, series_name="$H(c)$")
			plot_condit_entropy = PlotFunc(_entropy_of_ba_conditional_pmf_wrapper, param_dict=param_dict, x_min=0.0, x_max=5.0, x_interval=0.2, series_name="$H(c \\mid m)$")

			# Hang on to it.
			funcs_to_plot.append(plot_ba)
			funcs_to_plot.append(plot_entropy)
			funcs_to_plot.append(plot_condit_entropy)

	# Plot.
	from func_plotting import plot_funcs
	plot_funcs(funcs_to_plot, title="Blahut-Arimoto: $\\beta\\  vs. \\Pr(code \\mid message)$", x_label="$\\beta$")

# ----------
# -- Main --
# ----------

def main():
	plotting = True

	if plotting:
		# Make plots
		make_ba_plot()
	else:
		# Just print out the coding distribution.
		pmf_code, coding_distr = blahut_arimoto(beta=0.0)
		print_pmf(pmf_code)
		print_coding_distr(coding_distr)
		print "beta =", beta
	
if __name__ == "__main__":
	main()