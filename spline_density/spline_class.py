import numpy as np
from spline_density.utils import gaussian_quadrature
import numbers

class Spline:
    def __init__(self, degree, knots, coeffs=None):
        self.knots = np.asarray(knots, dtype=float)
        self.degree = int(degree)

        self.dim = (self.degree + 1) + len(self.knots)
        if coeffs is None:
            self.coeffs = np.zeros(self.dim, dtype=float)
        else:
            coeffs = np.asarray(coeffs, dtype=float)
            if coeffs.size == self.dim:
                self.coeffs = coeffs
            else:
                raise ValueError(
                    f"Expected {self.dim} coefficients (degree+1+n_knots), got {coeffs.size}"
                )
    
    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        shape = x.shape
        x = x.ravel()
        
        res = self.coeffs @ self.spline_basis(x)
        
        if shape == ():
            return res[0]
        else:
            return res.reshape(shape)
    
    def spline_basis(self, x):
        x = np.asarray(x, dtype=float)
        powers = np.zeros((self.dim, x.size), dtype=float)
        
        powers[:(self.degree+1),:] = np.vstack([x**k for k in range(self.degree+1)])
        
        if len(self.knots) > 0:
            diffs = x[None, :] - self.knots[:, None]
            diffs = np.maximum(diffs, 0.0)
            powers[(self.degree+1):,:] = diffs**self.degree
        
        return powers
        
    def gradient_coeffs(self, x):
        return self.spline_basis(x)
    
    def derivative(self, q = 1):
        if not isinstance(q, int):
            raise TypeError("q needs to be an integer")
        if q < 0:
            raise TypeError("q needs to be non-negative")
        if q == 0:
            return self
        if q > 1:
            return self.derivative(q = q-1).derivative()
        
        coeffs = self.coeffs[1:] * np.minimum(np.arange(1, self.dim), self.degree)
        
        return Spline(self.degree-1, self.knots, coeffs=coeffs)
    
    def integral(self, c = 0):
        coeffs = np.concatenate(([c], self.coeffs)) * np.concatenate(([1.], 1./np.arange(1, self.degree+2), np.ones(self.knots.size)/(self.degree+1)))
        
        return Spline(self.degree+1, self.knots, coeffs=coeffs)
    
    def __add__(self, other):
        if np.isscalar(other):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += other
            return Spline(degree=self.degree, knots=self.knots, coeffs=new_coeffs)
        
        if not isinstance(other, Spline):
            return NotImplemented
        if self.degree != other.degree:
            raise ValueError("To sum two splines, their degrees must be equal.")

        new_knots = np.sort(np.unique(np.concatenate([self.knots, other.knots])))

        degree = self.degree
        new_dim = (degree + 1) + len(new_knots)

        
        self_coeffs_new = np.zeros(new_dim)
        self_coeffs_new[:degree+1] = self.coeffs[:degree+1]
        for i, t in enumerate(self.knots):
            new_index = np.where(new_knots == t)[0][0]
            self_coeffs_new[degree+1 + new_index] = self.coeffs[degree+1 + i]

        other_coeffs_new = np.zeros(new_dim)
        other_coeffs_new[:degree+1] = other.coeffs[:degree+1]
        for i, t in enumerate(other.knots):
            new_index = np.where(new_knots == t)[0][0]
            other_coeffs_new[degree+1 + new_index] = other.coeffs[degree+1 + i]

        sum_coeffs = self_coeffs_new + other_coeffs_new
        return Spline(degree, new_knots, coeffs=sum_coeffs)
    def __neg__(self):
        return Spline(degree=self.degree, knots=self.knots, coeffs=-self.coeffs)
    def __sub__(self, other):
        return self.__add__(-other)
    def __rsub__(self, other):
        return (-self).__radd__(other)
    
    def __mul__(self, scalar):
        if not np.isscalar(scalar):
            return NotImplemented
        return Spline(self.degree, self.knots, coeffs=self.coeffs * float(scalar))
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    def __truediv__(self, scalar):
        if not np.isscalar(scalar):
            return NotImplemented
        return Spline(self.degree, self.knots, coeffs=self.coeffs / float(scalar))
    

class SplineDistribution():
    def __init__(self, h, L, U, quadrature_n = None, clip_lim = 1e9, old_gamma = 0., new_gamma = 0.):        
        self.h = h
        self.h_logpdf = lambda z: np.clip(h(z), -clip_lim, clip_lim)
        self.L = L
        self.U = U
        
        if quadrature_n is None:
            self.quadrature_n = 2*self.h.degree
        else:
            self.quadrature_n = quadrature_n
        
        self.t = np.concatenate(([L], self.h.knots, [U]))
        
        a = gaussian_quadrature(self.quadrature_n, lambda z: np.exp(self.h_logpdf(z)), self.t[:-1], self.t[1:])
        cum_a = np.concatenate(([0.], a.cumsum()))
        
        self.log_density = lambda z: self.h_logpdf(z) - np.log(cum_a[-1])
        
        self.t_cum_density = cum_a / cum_a[-1]
        
        self.old_gamma = old_gamma
        self.new_gamma = new_gamma
    
    @property
    def gamma(self):
        return (self.new_gamma-self.old_gamma) / (1.-self.old_gamma)
    
    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        shape = x.shape
        x = x.ravel()
        
        if self.gamma == 0.:
            res = np.where((x >= self.L) & (x <= self.U), self.log_density(x), -np.inf)
        else:
            res = np.where((x >= self.L) & (x <= self.U), np.log(self.pdf(x)), -np.inf)
        
        if shape == ():
            return res[0]
        else:
            return res.reshape(shape)
    
    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        shape = x.shape
        x = x.ravel()
        
        res = (1-self.gamma)*np.exp(self.log_density(x)) + self.gamma/(self.U-self.L)
        res = np.where((x >= self.L) & (x <= self.U), res, 0.)
        
        if shape == ():
            return res[0]
        else:
            return res.reshape(shape)
    
    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        shape = x.shape
        x = x.ravel()
        
        interval = np.searchsorted(self.t, x)-1
        increment = gaussian_quadrature(self.quadrature_n, lambda z: np.exp(self.log_density(z)), self.t[interval], x)
        
        res = (1-self.gamma)*(self.t_cum_density[interval] + increment) + self.gamma*(x-self.L)/(self.U-self.L)
        res = np.where(x > self.L, np.where(x < self.U, res, 1.), 0.)
        
        if shape == ():
            return res[0]
        else:
            return res.reshape(shape)
    
    def ppf(self, q, tol=1e-10, max_iterations=1000):
        q = np.asarray(q, dtype=float)
        shape = q.shape
        q = q.ravel()
        
        res = np.empty_like(q)

        mask_low = (q <= 0.)
        mask_high = (q >= 1.)
        mask_mid = (~mask_low) & (~mask_high)

        res[mask_low] = self.L
        res[mask_high] = self.U

        if np.any(mask_mid):
            q_mid = q[mask_mid]
            lo = np.full_like(q_mid, self.L)
            hi = np.full_like(q_mid, self.U)

            for _ in range(max_iterations):
                mid = 0.5 * (lo + hi)
                cdf_mid = self.cdf(mid)

                go_right = cdf_mid < q_mid
                lo[go_right] = mid[go_right]
                hi[~go_right] = mid[~go_right]

                if np.max(hi - lo) < tol:
                    break

            res[mask_mid] = 0.5 * (lo + hi)

        if shape == ():
            return res[0]
        else:
            return res.reshape(shape)
    
    def rvs(size=1, rng=None):
        if rng is None or isinstance(rng, numbers.Integral):
            rng = np.random.default_rng(rng)
        elif not isinstance(rng, np.random.Generator):
            raise TypeError("`rng` must be None, an integer, or a numpy.random.Generator.")
        
        return self.ppf(rng.uniform(0.0, 1.0, size=size))