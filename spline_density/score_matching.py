import numpy as np
from spline_density.spline_class import Spline, SplineDistribution
from spline_density.utils import construct_G

def score_matching(
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
    
    if theta0 is None:
        theta0 = np.zeros(k + degree + 1)
    else:
        theta0 = np.asarray(theta0, dtype=float)
        if theta0.size != k + degree + 1:
            raise ValueError(f"theta0 has the wrong size {theta0.size}. Should be {k + degree + 1}.")
            
    ## Define variables ##
    theta = theta0.copy()
    coeffs_theta = theta
    h = Spline(degree=degree, knots=t[1:-1], coeffs=coeffs_theta)
    
    G_0 = construct_G(degree, t, 0)
    G_q = construct_G(degree, t, q)
    
    ## Iterate ##
    iterations = 0
    ell = -np.inf
    converged = False
    while iterations < max_iterations:
        de_h = h.derivative()
        
        old_ell = ell
        ell = - ((1-gamma)*de_h(sample) + 0.5 * (1-gamma) * h(sample)**2).mean() - 0.5*gamma/(U-L)*(theta @ G_0 @ theta) - 0.5*lambda_n*(theta @ G_q @ theta)
        
        eps = (old_ell - ell)/max(1., abs(old_ell)) if np.isfinite(old_ell) else eps_conv + 1.
        if 0 <= eps < eps_conv:
            converged = True
            break

        h_basis = h.spline_basis(sample)
        de_h_basis = de_h.spline_basis(sample)

        gradient = - ((1-gamma) * np.vstack([np.zeros((1, de_h_basis.shape[1])), de_h_basis]) * np.minimum(np.arange(k + degree + 1), degree)[:, None] \
                    + (1-gamma) * h_basis * h(sample)[None, :]).mean(axis=1) - gamma/(U-L)*(theta @ G_0) - lambda_n*(theta @ G_q)
        Hessian = - (1-gamma) * h_basis @ h_basis.T / sample.size - gamma/(U-L)*G_0 - lambda_n*G_q
            
        theta -= zeta * np.linalg.solve(Hessian, gradient)
        
        coeffs_theta = theta
        h.coeffs = coeffs_theta

        iterations += 1
    
    distr = SplineDistribution(h.integral(), L, U)
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