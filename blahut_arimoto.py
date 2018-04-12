# Python imports.
import math
from collections import defaultdict

# ---------------------
# -- Misc. Functions --
# ---------------------

def distance(x, x_tilde):
	total_dist = 0.5
	for i, char in enumerate(x[1:]):
		if x_tilde[i-1] != char:
			total_dist += 1

	return total_dist


def _compute_denominator(x_tilde, p_of_x_tilde, all_x, beta):
	'''
	Args:
		x_tilde (str)
		p_of_x_tilde (dict)
		all_x (list)
		beta (float)

	Returns:
		(float)
	'''
	return sum([p_of_x_tilde[x_tilde] * math.exp(- beta * distance(x_tilde, x)) for x in all_x])

def init_p_of_x(all_x):
	'''
	Args:
		all_x (list)

	Returns:
		(dict)
	'''

	new_p_x = defaultdict(float)
	
	new_p_x["11"] = 0.5
	new_p_x["10"] = 0.1
	new_p_x["01"] = 0.1
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
		print "  code:", x_tilde
		new_p_of_x_tilde[x_tilde] = sum([p_of_x[x] * coding_distribution[x][x_tilde] for x in p_of_x.keys()])
		print "\tp(code):", new_p_of_x_tilde[x_tilde]
		for x in p_of_x.keys():
			print "\t  x", x
			print "\t    p(code | x):", coding_distribution[x][x_tilde]

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

	all_x = p_of_x.keys()

	for x in coding_distribution.keys():
		for x_tilde in coding_distribution.values()[0].keys():

			numerator = p_of_x_tilde[x_tilde] * math.exp(-beta * distance(x, x_tilde))
			denominator = _compute_denominator(x_tilde, p_of_x_tilde, all_x, beta)

			new_coding_distribution[x][x_tilde] = float(numerator) / denominator

	# print new_coding_distribution

	return new_coding_distribution


def main():

	# Beta.
	beta = 0.01

	# Make message alphabet.
	all_x = ["00","01","10","11"]

	# Make coding alphabet.
	all_x_tilde = ["0","1"]

	# Init pmfs.
	p_of_x = init_p_of_x(all_x)
	p_of_x_tilde = init_p_of_x_tilde(all_x_tilde)
	coding_distribution = init_coding_distribution(all_x, all_x_tilde)

	# Blahut.
	for i in range(3):
		print "Round", i
		p_of_x_tilde = compute_prob_of_codes(p_of_x, coding_distribution, beta=beta)
		coding_distribution = compute_coding_distribution(p_of_x, p_of_x_tilde, coding_distribution, beta=beta)

	print "\n~~~ DONE ~~~"
	print "p(code):", p_of_x_tilde
	print "p(code | x):", coding_distribution

if __name__ == "__main__":
	main()