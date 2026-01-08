import numpy as np
from spline_density.spline_class import Spline, SplineDistribution
from spline_density.utils import construct_G, gaussian_quadrature, tartaglia_line

def MLE_density(
    sample,
    degree,
    q,
    lambda_n,
    L = None,
    U = None,
    k = None,
    t = None,
    theta0 = None,
    zeta = 0.1,
    gamma = 0.,
    quadrature_n = None,
    clip_lim = 50,
    eps_conv = 1e-6,
    max_iterations = 1000,
    output_stats = False,
    **kwargs,
    ):
    
    ## Data validation ##
    sample = np.asarray(sample, dtype=float).ravel()
    degree = int(degree)
    q = int(q)
    lambda_n = float(lambda_n)
    
    if t is None:
        if k is None:
            raise ValueError("You must specify at least one between 't' (knots) or 'k' (number of intervals).")
        if L is None:
            L = min(sample)
        if U is None:
            U = max(sample)
        t = np.linspace(L, U, k+2)
    else:
        if k is not None and len(t)-2 != k:
            raise ValueError("The specified k and t are incompatible.")
        k = len(t)-2
        if L is not None and t[0] != L:
            raise ValueError("The specified L and t are incompatible.")
        L = t[0]
        if U is not None and t[-1] != U:
            raise ValueError("The specified U and t are incompatible.")
        U = t[-1]
    if quadrature_n is None:
        quadrature_n = 2*degree
    
    if theta0 is None:
        theta0 = np.zeros(k + degree)
    else:
        theta0 = np.asarray(theta0, dtype=float)
        if theta0.size != k + degree:
            raise ValueError(f"theta0 has the wrong size {theta0.size}. Should be {k + degree}.")
            
    ## Define variables ##
    theta = theta0.copy()
    coeffs_theta = np.concatenate(([0.], theta))
    h = Spline(degree=degree, knots=t[1:-1], coeffs=coeffs_theta)
    
    G_q = construct_G(degree, t, q)[1:,1:]
    grad_int = construct_G(degree, t, 0)[0,1:]
    
    tartaglia_coeffs_I = tartaglia_line(degree)
    
    ## Iterate ##
    iterations = 0
    ell = -np.inf
    converged = False
    while iterations < max_iterations:
        a = np.zeros((k+1, 2*degree+1), dtype=float)
        for j in range(2*degree+1):
            f = lambda x: x**j * np.exp(np.clip(h(x), -clip_lim, clip_lim))
            a[:,j] = gaussian_quadrature(quadrature_n, f, t[:-1], t[1:])
        
        I0 = a[:,0].sum()
        
        I1 = np.zeros(k + degree, dtype=float)
        for i in range(1,degree+1):
            I1[i-1] = a[:,i].sum()
        for i in range(k):
            power_coeffs = (-t[i+1]) ** np.arange(degree, -1, -1) * tartaglia_coeffs_I
            I1[degree+i] = (a[(i+1):,0:(degree+1)].sum(axis=0) * power_coeffs).sum()
        
        I2 = np.zeros((k + degree, k + degree), dtype=float)
        I2[:degree, :degree] = a.sum(axis=0)[np.arange(1,degree+1)[None,:]+np.arange(1,degree+1)[:,None]]
        for i in range(1,degree+1):
            for j in range(k):
                power_coeffs = (-t[j+1]) ** np.arange(degree, -1, -1) * tartaglia_coeffs_I
                integral = (a[(j+1):,i:(degree+i+1)].sum(axis=0) * power_coeffs).sum()
                I2[i-1,degree+j] = integral
                I2[degree+j,i-1] = integral
        for i in range(k):
            for j in range(i, k): # note j >= i
                power_coeffs_i = (-t[i+1]) ** np.arange(degree, -1, -1) * tartaglia_coeffs_I
                power_coeffs_j = (-t[j+1]) ** np.arange(degree, -1, -1) * tartaglia_coeffs_I
                coeffs_prod_flipped = (power_coeffs_i[None,:] * power_coeffs_j[:, None])[:, ::-1]
                prod_coeffs = np.array([coeffs_prod_flipped.diagonal(offset=k).sum() for k in range(degree, -degree-1, -1)])
                
                integral = (a[(j+1):,:].sum(axis=0) * prod_coeffs).sum()
                I2[degree+i,degree+j] = integral
                I2[degree+j,degree+i] = integral
    
        old_ell = ell
        h_int = h.integral()
        ell = (1-gamma)*h(sample).mean() + gamma/(U-L)*(h_int(U)-h_int(L)) - np.log(I0) - 0.5*lambda_n*(theta @ G_q @ theta)
        
        eps = (old_ell - ell)/max(1., abs(old_ell)) if np.isfinite(old_ell) else eps_conv + 1.
        if 0 <= eps < eps_conv:
            converged = True
            break
        
        gradient = (1-gamma)*h.spline_basis(sample)[1:,:].mean(axis=1) + gamma/(U-L)*grad_int - I1/I0 - lambda_n*(theta @ G_q)
        Hessian = (I1[None,:] * I1[:, None] - I2) / I0**2 - lambda_n*G_q
        
        theta -= zeta * np.linalg.solve(Hessian, gradient)
        
        coeffs_theta = np.concatenate(([0.], theta))
        h.coeffs = coeffs_theta
    
        iterations += 1
    
    distr = SplineDistribution(h, L, U)
    if output_stats:
        stats = dict()
        stats["iterations"] = iterations
        stats["ell"] = ell
        stats["old_ell"] = old_ell
        stats["last_eps"] = eps
        stats["converged"] = converged
        return distr, stats
    else:
        return distr