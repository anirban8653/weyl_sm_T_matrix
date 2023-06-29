import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numba import jit, prange, njit

kernum = 16;
norb = 4;
hrfile = np.loadtxt('4_weyl_hr.dat')


def t(a, b, c, d, e):
    idx = np.where((hrfile[:, 0] == a) & (hrfile[:, 1] == b) & (hrfile[:, 2] == c) & (hrfile[:, 3] == d) & (hrfile[:, 4] == e))[0]
    if len(idx) > 0:
        return hrfile[idx, 5][0] + 1j * hrfile[idx, 6][0]
    else:
        return None

kkm = 50
afb = 0.005
x = 0.0

def ham(kx, ky, kz):
    result = np.zeros((norb, norb), dtype=complex)
    for m in range(1,norb+1):
        for n in range(1,norb+1):
            temp = 0
            for cx in range(-1, 2):
                for cy in range(-1, 2):
                    for cz in range(-1, 2):
                        value = t(cx, cy, cz, m, n)
                        if value is not None:
                            temp += value * np.exp(1j * (cx * kx + cy * ky + cz * kz))
            result[m-1][n-1] = temp / (np.sqrt(2 * np.pi) ** 3)
    return result


#%%
def Gk(kx, ky, kz):
    H = x + 1j * afb
    return np.linalg.inv(H * np.eye(norb) - ham(kx, ky, kz))

def Gkr(kx, ry, kz):
    result = np.zeros((norb, norb), dtype=complex)
    for ky in np.linspace(-np.pi, np.pi, kkm + 1)[:-1]:
        temp = Gk(kx, ky, kz)
        result += np.exp(1j * (ky * ry)) * temp
    return result / kkm

Vimp = 100.0
Himp = Vimp * np.eye(norb)

def Tmatrix(kx, kz):
    G = Gkr(kx, 0.0, kz)
    return np.linalg.inv(np.eye(norb) - Himp @ G) @ Himp

def Gfull(kx, kz):
    Gkr_kx_0 = Gkr(kx, 0.0, kz)
    Gkr_kx_minus1 = Gkr(kx, -1, kz)
    Gkr_kx_plus1 = Gkr(kx, 1, kz)
    return Gkr_kx_0 + Gkr_kx_minus1 @ Tmatrix(kx, kz) @ Gkr_kx_plus1

def calculate_A(kx, kz):
    trace = np.trace(Gfull(kx, kz))
    return [kx, kz, -1.0 / np.pi * np.imag(trace)]





A = Parallel(n_jobs=kernum)(delayed(calculate_A)(kx, kz) for kx in np.linspace(-np.pi / 2, np.pi / 2, kkm) for kz in np.linspace(-np.pi / 2, np.pi / 2, kkm))

#%%
A = np.array(A)
X = A[:, 0]
Y = A[:, 1]
Z = A[:, 2]

plot = plt.imshow(Z.reshape(kkm,kkm).T,extent=(np.amin(X), np.amax(X), np.amin(Y), np.amax(Y)), interpolation='bilinear',cmap='Blues')

plt.colorbar(plot)
plt.show()
np.savetxt("calculated_A.dat", A)