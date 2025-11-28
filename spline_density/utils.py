import numpy as np
from math import factorial as fac
from math import comb

def construct_G(degree, t, q):
    k = t.size-2
    tartaglia_coeffs = tartaglia_line(degree-q)
    
    G = np.zeros((k+degree+1, k+degree+1), dtype=float)
    for i in range(q, degree+1):
        for j in range(i, degree+1):
            integral = (fac(i)/fac(i-q))*(fac(j)/fac(j-q))*(t[-1]**(i+j-2*q+1) - t[0]**(i+j-2*q+1))/(i+j-2*q+1)
            G[i,j] = integral
            G[j,i] = integral
    for i in range(q, degree+1):
        for j in range(k):
            power_coeffs = (-t[j+1]) ** np.arange(degree-q, -1, -1) * tartaglia_coeffs
            power_range = np.arange(i-q+1, i+degree-2*q+2)
            integral = (fac(i)/fac(i-q))*(fac(degree)/fac(degree-q)) * (power_coeffs * (t[-1]**power_range - t[j+1]**power_range) / power_range).sum()
            G[i, degree+1+j] = integral
            G[degree+1+j, i] = integral
    for i in range(k):
        for j in range(i, k):
            power_coeffs_i = (-t[i+1]) ** np.arange(degree-q, -1, -1) * tartaglia_coeffs
            power_coeffs_j = (-t[j+1]) ** np.arange(degree-q, -1, -1) * tartaglia_coeffs
            coeffs_prod_flipped = (power_coeffs_i[None,:] * power_coeffs_j[:, None])[:, ::-1]
            prod_coeffs = np.array([coeffs_prod_flipped.diagonal(offset=k).sum() for k in range(degree-q, -(degree-q)-1, -1)])
            power_range = np.arange(1, 2*degree-2*q+2)

            # note j >= i
            integral = (fac(degree)/fac(degree-q))**2 * (prod_coeffs * (t[-1]**power_range - t[j+1]**power_range) / power_range).sum()
            G[degree+1+i,degree+1+j] = integral
            G[degree+1+j,degree+1+i] = integral
    
    return G

def gaussian_quadrature(n, f, L, U):
    L = np.asarray(L)
    U = np.asarray(U)

    monodim = L.shape == () and U.shape == ()

    L = L.ravel()
    U = U.ravel()
    
    if L.size > 1 and U.size > 1 and L.size != U.size:
        raise ValueError("Incorrect sizes of U and L")
    
    if L.size == 1:
        L = np.array([L[0] for _ in range(U.size)])
    elif U.size == 1:
        U = np.array([U[0] for _ in range(L.size)])
    
    x, w = np.polynomial.legendre.leggauss(n)
    t = (0.5 * (x + 1))[None, :] * (U - L)[:, None] + L[:, None]
    w_scaled = (0.5 * (U - L))[:, None] * w[None, :]
    res = np.sum(w_scaled * f(t), axis=1)

    if monodim:
        return res[0]
    else:
        return res
        
def tartaglia_line(i):
    return np.array([comb(i, k) for k in range(i + 1)], dtype=int)