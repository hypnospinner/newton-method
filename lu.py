import math
import operator
import numpy as np

# SWAP OPERATIONS

# swap rows


def swapr(matrix, fr, to):
    ops = 0
    if fr != to:
        copy = np.copy(matrix)
        matrix[fr], matrix[to] = copy[to],  copy[fr]
        ops = 1
    return matrix, ops

# swap columns


def swapc(matrix, fr, to):
    ops = 0
    if(fr != to):
        copy = np.copy(matrix)
        matrix[::, fr], matrix[::, to] = copy[::, to], copy[::, fr]
        ops = 1
    return matrix, ops

# swap col and row at the same time


def swapcr(matrix, frr, tor, frc, toc):
    matrix, ops_1 = swapr(matrix, frr, tor)
    matrix, ops_2 = swapc(matrix, frc, toc)
    return matrix, ops_1 + ops_2


def pivot(L, U, P, Q, d):
    absU = abs(U)

    if absU[d:, d:].sum() < 1e-10:
        return False, L, U, P, Q

    i, j = np.where(absU[d:, d:] == absU[d:, d:].max())
    i[0] += d
    j[0] += d

    L, opsL = swapcr(L, i[0], d, j[0], d)
    U, opsU = swapcr(U, i[0], d, j[0], d)
    P, opsP = swapr(P, i[0], d)
    Q, opsQ = swapc(Q, j[0], d)

    return True, L, U, P, Q, opsL + opsP + opsQ + opsU


def factorize2(matrix):
    l = len(matrix)
    U = np.copy(matrix)
    L = np.zeros((l, l))
    P, Q = np.eye(l), np.eye(l)
    ops = 0

    for i in range(l-1):

        success, L, U, P, Q, opsPivot = pivot(L, U, P, Q, i)

        ops += opsPivot

        if success == False:
            break

        T = np.eye(l)

        for k in range(i+1, l):
            L[k, i] = U[k, i] / U[i, i]
            T[k, i] = (-1) * L[k, i]
            ops += 2

        U = np.dot(T, U)
        ops += math.pow(l, 3)

    L = L + np.eye(l)
    ops += math.pow(l, 2)

    return L, U, P, Q, ops


def solveSLAE(matrix, b):
    L, U, P, Q, ops = factorize2(matrix)
    x = np.zeros(len(b))
    y = np.zeros(len(b))

    pb = np.dot(P, b)
    l = len(pb)
    ops += math.pow(l, 3)

    # Ly = Pb
    for k in range(l):
        y[k] = pb[k]

        for j in range(k):
            y[k] -= y[j] * L[k, j]

        ops += k

    # Ux = y
    for k in range(l-1, -1, -1):
        x[k] = y[k]

        for j in range(k+1, l):
            x[k] -= x[j] * U[k, j]

        x[k] /= U[k, k]
        ops += k

    return np.dot(Q, x), ops
