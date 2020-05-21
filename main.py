import numpy as np
import operator
import math
import lu
import timeit as tmr


def F(x):
    f = np.mat([
        math.cos(x[1] * x[0]) - math.exp(-3 * x[2]) + x[3] * x[4] ** 2 -
        x[5] - math.sinh(2 * x[7]) * x[8] + 2 * x[9] + 2.000433974165385440,
        math.sin(x[1] * x[0]) + x[2] * x[8] * x[6] - math.exp(-x[9] + x[5]
                                                              ) + 3 * x[4] ** 2 - x[5] * (x[7] + 1) + 10.886272036407019994,
        x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] -
        x[7] + x[8] - x[9] - 3.1361904761904761904,
        2 * math.cos(-x[8] + x[3]) + x[4] / (x[2] + x[0]) - math.sin(x[1]
                                                                     ** 2) + math.cos(x[6] * x[9]) ** 2 - x[7] - 0.1707472705022304757,
        math.sin(x[4]) + 2 * x[7] * (x[2] + x[0]) - math.exp(-x[6] * (-x[9] + x[5])
                                                             ) + 2 * math.cos(x[1]) - 1.0 / (-x[8] + x[3]) - 0.3685896273101277862,
        math.exp(x[0] - x[3] - x[8]) + x[4] ** 2 / x[7] + math.cos(3 *
                                                                   x[9] * x[1]) / 2 - x[5] * x[2] + 2.0491086016771875115,
        x[1] ** 3 * x[6] - math.sin(x[9] / x[4] + x[7]) + (x[0] - x[5]) *
        math.cos(x[3]) + x[2] - 0.7380430076202798014,
        x[4] * (x[0] - 2 * x[5]) ** 2 - 2 * math.sin(-x[8] + x[2]) + 0.15e1 *
        x[3] - math.exp(x[1] * x[6] + x[9]) + 3.5668321989693809040,
        7 / x[5] + math.exp(x[4] + x[3]) - 2 * x[1] * x[7] * x[9] *
        x[6] + 3 * x[8] - 3 * x[0] - 8.4394734508383257499,
        x[9] * x[0] + x[8] * x[1] - x[7] * x[2] +
        math.sin(x[3] + x[4] + x[5]) * x[6] - 0.78238095238095238096])

    return f


def J(x):
    j = np.mat([
        [-x[1] * math.sin(x[1] * x[0]), -x[0] * math.sin(x[1] * x[0]), 3 * math.exp(-3 * x[2]), x[4] ** 2, 2 * x[3] * x[4],
         -1, 0, -2 * math.cosh(2 * x[7]) * x[8], -math.sinh(2 * x[7]), 2],
        [x[1] * math.cos(x[1] * x[0]), x[0] * math.cos(x[1] * x[0]), x[8] * x[6], 0, 6 * x[4],
         -math.exp(-x[9] + x[5]) - x[7] - 1, x[2] * x[8], -x[5], x[2] * x[6], math.exp(-x[9] + x[5])],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [-x[4] / (x[2] + x[0]) ** 2, -2 * x[1] * math.cos(x[1] ** 2), -x[4] / (x[2] + x[0]) ** 2, -2 * math.sin(-x[8] + x[3]),
         1.0 / (x[2] + x[0]), 0, -2 * math.cos(x[6] * x[9]) *
         x[9] * math.sin(x[6] * x[9]), -1,
            2 * math.sin(-x[8] + x[3]), -2 * math.cos(x[6] * x[9]) * x[6] * math.sin(x[6] * x[9])],
        [2 * x[7], -2 * math.sin(x[1]), 2 * x[7], 1.0 / (-x[8] + x[3]) ** 2, math.cos(x[4]),
         x[6] * math.exp(-x[6] * (-x[9] + x[5])), -(x[9] - x[5]) *
            math.exp(-x[6] * (-x[9] + x[5])), 2 * x[2] + 2 * x[0],
            -1.0 / (-x[8] + x[3]) ** 2, -x[6] * math.exp(-x[6] * (-x[9] + x[5]))],
        [math.exp(x[0] - x[3] - x[8]), -1.5 * x[9] * math.sin(3 * x[9] * x[1]), -x[5], -math.exp(x[0] - x[3] - x[8]),
         2 * x[4] / x[7], -x[2], 0, -x[4] ** 2 /
         x[7] ** 2, -math.exp(x[0] - x[3] - x[8]),
            -1.5 * x[1] * math.sin(3 * x[9] * x[1])],
        [math.cos(x[3]), 3 * x[1] ** 2 * x[6], 1, -(x[0] - x[5]) * math.sin(x[3]),
         x[9] / x[4] ** 2 * math.cos(x[9] / x[4] + x[7]),
            -math.cos(x[3]), x[1] ** 3, -math.cos(x[9] / x[4] + x[7]), 0, -1.0 / x[4] * math.cos(x[9] / x[4] + x[7])],
        [2 * x[4] * (x[0] - 2 * x[5]), -x[6] * math.exp(x[1] * x[6] + x[9]), -2 * math.cos(-x[8] + x[2]), 1.5,
         (x[0] - 2 * x[5]) ** 2, -4 * x[4] * (x[0] - 2 * x[5]), -
            x[1] * math.exp(x[1] * x[6] + x[9]), 0,
            2 * math.cos(-x[8] + x[2]),
            -math.exp(x[1] * x[6] + x[9])],
        [-3, -2 * x[7] * x[9] * x[6], 0, math.exp(x[4] + x[3]), math.exp(x[4] + x[3]),
         -7.0 / x[5] ** 2, -2 * x[1] * x[7] * x[9], -2 * x[1] * x[9] * x[6], 3, -2 * x[1] * x[7] * x[6]],
        [x[9], x[8], -x[7], math.cos(x[3] + x[4] + x[5]) * x[6], math.cos(x[3] + x[4] + x[5]) * x[6],
         math.cos(x[3] + x[4] + x[5]) * x[6], math.sin(x[3] + x[4] + x[5]), -x[2], x[1], x[0]]])

    return j


def coefs(x):
    return np.array(-1 * F(x))[0]


def checkAccuracy(x, eps):
    c = coefs(x)

    for i in range(len(c)):
        if abs(coefs(x)[i]) > eps:
            return False

    return True


def newton_step(x):
    j = np.array(J(x))
    b = coefs(x)

    tx, ops = lu.solveSLAE(j, b)
    return j, tx, ops


def newton_mod_step(P, Q, L, U, x):
    ops = 0
    b = coefs(x)

    x = np.zeros(len(b))
    y = np.zeros(len(b))

    pb = np.dot(P, b)
    ops += len(b)

    for k in range(len(y)):
        y[k] = pb[k]
        ops += 1

        for j in range(k):
            y[k] -= y[j] * L[k, j]
            ops += 1

    for k in range(len(x)-1, -1, -1):
        x[k] = y[k]
        for j in range(k+1, len(x)):
            x[k] -= x[j] * U[k, j]
            ops += 1

        x[k] /= U[k, k]
        ops += 2

    stepr = np.dot(Q, x)

    # we cal dot product 2 times which takes n^3 operations
    ops += 2 * math.pow(len(x), 3)

    return stepr, ops


def appr(F, J, x, eps=1e-10):
    pr = np.copy(x)
    iters = 0
    ops = 0

    st = tmr.default_timer()

    while iters < 20000:
        iters += 1
        _, tr, stepOps = newton_step(pr)
        ops += stepOps
        nr = tr + pr

        if np.linalg.norm(nr - pr) < eps:
            accurate = checkAccuracy(nr, eps)
            if accurate:
                fin = tmr.default_timer()
                return True, nr, iters, ops, fin - st
            else:
                break

        pr = np.copy(nr)

    # this can mean 2 things: we performed all iteration steps and failed or
    # one of the functions is to great based on next and previous roots delta
    return False, None, None, None, None


def mod_appr(F, J, x, eps=1e-10, lim=1):
    pr = np.copy(x)
    iters = 0
    ops = 0
    L, U, P, Q = None, None, None, None

    st = tmr.default_timer()

    while iters < 20000:

        if iters < lim:
            T, tr, stepOps = newton_step(pr)
            ops += stepOps
        else:
            if L is None:
                L, U, P, Q, factSteps = lu.factorize2(T)
                ops += factSteps

            tr, stepOps = newton_mod_step(P, Q, L, U, pr)
            ops += stepOps

        nr = tr + pr

        if np.linalg.norm(nr - pr) < eps:
            accurate = checkAccuracy(nr, eps)
            if accurate:
                fin = tmr.default_timer()
                return True, nr, iters, ops, fin - st
            else:
                break

        pr = np.copy(nr)
        iters += 1

    return False, None, None, None, None


def hyprid_appr(F, J, x, eps=1e-10, multiple=1):
    pr = np.copy(x)
    iters = 0
    ops = 0
    L, U, P, Q = None, None, None, None

    st = tmr.default_timer()

    while iters < 20000:

        if iters % multiple == 0:
            T, tr, stepOps = newton_step(pr)
            ops += stepOps
        else:
            if L is None:
                L, U, P, Q, factSteps = lu.factorize2(T)
                ops += factSteps

            tr, stepOps = newton_mod_step(P, Q, L, U, pr)
            ops += stepOps

        nr = tr + pr

        if np.linalg.norm(nr - pr) < eps:
            accurate = checkAccuracy(nr, eps)
            if accurate:
                fin = tmr.default_timer()
                return True, nr, iters, ops, fin - st
            else:
                break

        pr = np.copy(nr)
        iters += 1

    return False, None, None, None, None


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# x = np.array([0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5])
x = np.array([0.5, 0.5, 1.5, -1.0, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5])

success, root, iterations, operations, dt = appr(F, J, x, 1e-4)

print("\nDEFAULT NEWTON METHOD\n")

if success:
    print(f"Root:\n{root}\n")
    print(f"Iterations: {iterations}\n")
    print(f"Operations: {operations}\n")
    print(f"Delta time: {dt}\n")
    print(f"F(root):\n{F(root)}\n")
else:
    print("Default method failed")
    
print("\nMODIFIED NEWTON METHOD\n")

try:
    success, root, iterations, operations, dt = mod_appr(F, J, x, 1e-4, 7)
    if success:
        print(f"Root:\n{root}\n")
        print(f"Iterations: {iterations}\n")
        print(f"Operations: {operations}\n")
        print(f"Delta time: {dt}\n")
        print(f"F(root):\n{F(root)}\n")
    else:
        print("Modified method failed")
except OverflowError:
    print("Modified method failed")

try:
    success, root, iterations, operations, dt = hyprid_appr(F, J, x, 1e-4)

    print("\nHYBRID NEWTON METHOD\n")

    if success:
        print(f"Root:\n{root}\n")
        print(f"Iterations: {iterations}\n")
        print(f"Operations: {operations}\n")
        print(f"Delta time: {dt}\n")
        print(f"F(root):\n{F(root)}\n")
    else:
        print("Hybrid method failed")
except OverflowError:
    print("Hybrid method failed")
