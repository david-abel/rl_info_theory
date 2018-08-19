# Python imports.
import math
import os

# -------------------
# -- Entropy Funcs --
# -------------------

def l1_distance(pmf_p, pmf_q):
    '''
    Args:
        pmf_p (dict)
        pmf_q (dict)

    Returns:
        (float)
    '''
    total_l1 = 0.0
    for x in pmf_p.keys():
        total_l1 += abs(pmf_p[x] - pmf_q[x])
    return total_l1

def kl(pmf_p, pmf_q):
    '''
    Args:
        pmf_p (dict)
        pmf_q (dict)

    Returns:
        (float)
    '''
    kl_divergence = 0.0
    for x in pmf_p.keys():
        if pmf_p[x] > 0.0:  # Avoid division by zero.
            if pmf_q[x] == 0.0:
                return float('inf')

            kl_divergence += pmf_p[x] * math.log(pmf_p[x] / pmf_q[x], 2)

    return kl_divergence

def entropy(pmf):
    '''
    Args:
        pmf (dict)

    Returns:
        (float)
    '''
    total = 0
    for x in pmf.keys():
        if pmf[x] == 0:
            # Assume log_b 0 = 0.
            continue

        total -= pmf[x] * math.log(pmf[x], 2)

    return total


def conditional_entropy(pmf_y_given_x, pmf_x):
    '''
    Args:
        pmf_y_given_x
        pmf_x

    Returns:
        (float): H(Y | X)
    '''
    total = 0
    for x in pmf_x.keys():
        for y in pmf_y_given_x[x].keys():
            if pmf_y_given_x[x][y] == 0:
                continue
            total -= pmf_y_given_x[x][y] * pmf_x[x] * math.log(pmf_y_given_x[x][y], 2)

    return total

def mutual_info(pmf_x, pmf_y, pmf_x_given_y):
    '''
    Args:
        pmf_x (dict)
        pmf_y (dict)

    Returns:
        (float): I(X; Y) = H(X) - H(X | Y)
    '''
    # if entropy(pmf_x) < 0.0:
    #   print_pmf(pmf_x)
    return entropy(pmf_x) - conditional_entropy(pmf_x_given_y, pmf_y)

# ---------------------
# -- Print Functions --
# ---------------------

def print_pmf(pmf, name='p(x)'):
    '''
    Args:
        pmf (dict):
            Key=object
            Value=numeric (int/float)
    '''
    print "\n-- " + str(name) + " --"
    for x in pmf.keys():
        print "  " + str(name)[:-1] + " = " + str(x) + ") =", round(pmf[x], 3)
    print "----------\n"

def print_coding_pmf(coding_distr):
    '''
    Args:
        coding_distr (dict):
            Key=object
            Val=dict
                Key=object
                Val=numeric (int/float)
    '''
    print "\n-- Coding Distr --"
    for x in coding_distr.keys():
        print "x =", x
        for code in coding_distr[x]:
            print "  p(code = " + str(code) + " | x = " + str(x) + "):", round(coding_distr[x][code], 3)
        print
    print "--------------"

# --------------------
# -- Misc Functions --
# --------------------

def write_datum_to_file(file_name, datum, extra_dir=""):
    '''
    Summary:
        Writes @datum to extra_dir/file_name.
    '''
    if extra_dir != "" and not os.path.isdir(extra_dir):
        os.makedirs(extra_dir)
    out_file = open(os.path.join(extra_dir, file_name) + ".csv", "a+")
    out_file.write(str(datum) + ",")
    out_file.close()

def end_of_instance(file_name, extra_dir=""):
    out_file = open(os.path.join(extra_dir, str(file_name)) + ".csv", "a+")
    out_file.write("\n")
    out_file.close()

def clear_files(dir_name):
    '''
    Args:
        dir_name (str)

    Summary:
        Removes all csv files in @dir_name.
    '''
    for extension in ["iters", "times"]:
        dir_w_extension = os.path.join(dir_name, extension)  # , mdp_type) + ".csv"
        if not os.path.exists(dir_w_extension):
            os.makedirs(dir_w_extension)

        for mdp_type in ["vi", "vi-$\phi_{Q_d^*}$"]:
            if os.path.exists(os.path.join(dir_w_extension, mdp_type) + ".csv"):
                os.remove(os.path.join(dir_w_extension, mdp_type) + ".csv")

def write_datum(file_name, datum):
    '''
    Args:
        file_name (str)
        datum (object)
    '''
    out_file = open(file_name, "a+")
    out_file.write(str(datum) + ",")
    out_file.close()
