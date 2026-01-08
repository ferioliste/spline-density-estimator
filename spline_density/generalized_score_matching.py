import numpy as np
from spline_density.spline_class import Spline, SplineDistribution
from spline_density.utils import construct_G
 
def generalized_score_matching(
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
    phi_cap = 1,
    alpha = 2,
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
    
    G_q = construct_G(degree, t, q)
    
    phi = lambda x: np.minimum(np.minimum(x-L, U-x), phi_cap)
    de_phi = lambda x: np.where(x < min(L+phi_cap, (L+U)/2), 1., 0.) + np.where(x > max(U-phi_cap, (L+U)/2), -1., 0.)
    g = lambda x: x**alpha
    de_g = lambda x: alpha*x**(alpha-1)
    
    ## Iterate ##
    iterations = 0
    ell = -np.inf
    converged = False
    while iterations < max_iterations:
        de_h = h.derivative()
        
        old_ell = ell
        ell = - ( \
                g(phi(sample))*de_h(sample) \
                + de_phi(sample)*de_g(phi(sample))*h(sample) \
                + 0.5*g(phi(sample))*h(sample)**2 \
                ).mean() \
                - 0.5*lambda_n*(theta @ G_q @ theta)
        
        eps = (old_ell - ell)/max(1., abs(old_ell)) if np.isfinite(old_ell) else eps_conv + 1.
        if 0 <= eps < eps_conv:
            converged = True
            break

        h_basis = h.spline_basis(sample)
        de_h_basis = de_h.spline_basis(sample)

        gradient = - ( \
            np.vstack([np.zeros((1, de_h_basis.shape[1])), de_h_basis]) * np.minimum(np.arange(k+degree+1), degree)[:, None] * g(phi(sample))[None, :] \
            + h_basis * (h(sample) * g(phi(sample)))[None, :] \
            + h_basis * (de_phi(sample) * de_g(phi(sample)))[None, :] \
            ).mean(axis=1) - lambda_n*(theta @ G_q)
        Hessian = - (h_basis * g(phi(sample))[None, :]) @ h_basis.T / sample.size - lambda_n*G_q
            
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