import numpy as np
import numpy.testing as npt
import math
import random

def random_prob(card):
    ret = np.zeros(card)
    for i in range(len(ret)):
        ret[i] = random.random()
    s = sum(ret)
    for i in range(len(ret)):
        ret[i] = ret[i] / s
    return ret    

def random_prob2(x_card, y_card):
    ret = np.zeros((x_card,y_card))
    for x in range(x_card):
        for i in range(len(ret[x])):
            ret[x][i] = random.random()
        s = sum(ret[x])
        for i in range(len(ret[x])):
            ret[x][i] = ret[x][i] / s
    return ret

def kl(p, q):
    assert(p.size == q.size)
    d = 0.0
    for i, x in enumerate(p):
        # print "pi, qi =", p[i], q[i]
        if q[i] > 0.0:  # zero div
            d += p[i] * math.log(p[i] / q[i])
    return d

def sum_of_kl(p, q):
    d = 0.0
    for i, x in enumerate(p):
        for j, y in enumerate(x):
            if q[i][j] > 0.0:
                d += p[i][j] * math.log(p[i][j] / q[i][j])
    return d

def bottleneck(px, py_x, c_card, beta, thrs=0.0001):
    # px     P(X)   : S -> R, stationary distribution
    # py_x   P(Y|X) : S, A -> R
    # c_card |X-childa|

    # print "py_x =", py_x
    
    x_card = py_x.shape[0]
    y_card = py_x.shape[1]

    # print "|X|, |Y|, |C| =", x_card, y_card, c_card

    npt.assert_almost_equal(sum(px), 1.0)
    for x in range(x_card):
        npt.assert_almost_equal(sum(py_x[x]), 1.0)
            

    # Initial distribution doesn't matter
    pc_x = random_prob2(x_card, c_card) # np.full((x_card, c_card), 1.0 / c_card)  # P(Xchilda|X)
    pc = random_prob(c_card) # np.full(c_card, 1.0 / c_card) # P(Xchilda)
    py_c = random_prob2(c_card, y_card) # np.full((c_card, y_card), 1.0 / y_card) # P(Y|Xchilda)

    pc_x_bf = np.zeros((x_card, c_card))

    while (np.amax(pc_x - pc_x_bf) > thrs):
    # for t in range(2):
        # print np.amax(pc_x - pc_x_bf)
        pc_x_bf = np.copy(pc_x)

        # 1. Calculate p(c | x)
        for x in range(x_card):
            for c in range(c_card):
                d = kl(py_x[x], py_c[c])
                # print "KL =", d
                pc_x[x][c] = pc[c] * math.exp(-beta * d)
                # print "pc_x[", x, "][", c, "] =", pc_x[x][c]

            z_xb = sum(pc_x[x]) # Normalization constant
            # assert(z_xb > 0.0)
            for c in range(c_card):
                pc_x[x][c] = pc_x[x][c] / z_xb
                # print "pc_x[", x, "][", c, "] =", pc_x[x][c]

            npt.assert_almost_equal(sum(pc_x[x]), 1.0)

        # 2. Calculate p(c)
        pc = np.zeros(c_card)
        for c in range(c_card):
            for x in range(x_card):
                pc[c] += pc_x[x][c] * px[x]
                
        npt.assert_almost_equal(sum(pc), 1.0)

        # 3. Calculate p(y | c)
        py_c = np.zeros((c_card, y_card))
        for c in range(c_card):
            for x in range(x_card):
                x_c = pc_x[x][c] * px[x] / pc[c]
                for y in range(y_card):
                    py_c[c][y] += py_x[x][y] * x_c

            npt.assert_almost_equal(sum(py_c[c]), 1.0)
        # print "pc_x", pc_x
        # print "pc", pc
        # print "py_c", py_c

    return (pc_x, pc, py_c)

def fourstates():
    # State space grap
    # 0 - 1 - 2 - 3
    # Action set = right, left

    # state distribution
    # ssd = np.full(4, 1.0 / 4.0)
    ssd = np.array([0.25, 0.25, 0.25, 0.25])

    # Demonstrated policy
    # Policy needs to be stochastic to calculate
    policy = np.array([[0.1, 0.9],
                     [0.2, 0.8],
                     [0.8, 0.2],
                     [0.9, 0.1]])
    c_cards = [2, 3, 4, 5]  # Abstract state space size
    
    betas = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    for c_card in c_cards:
        for beta in betas:
            pc_x, pc, py_c = bottleneck(ssd, policy, c_card, beta)
            print "##################"
            print "(b, |C|) = (", beta, ",", c_card, ")"
            print "##################"
            print "P(C|X)", pc_x
            print "P(C)", pc
            print "P(Y|C)", py_c


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    fourstates()
