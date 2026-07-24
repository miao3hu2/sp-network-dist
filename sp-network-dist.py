import numpy as np
from __future__ import annotations
from numpy.typing import NDArray
from typing import Callable

_ftype = np.float64
_ctype = np.complex128

class EmpiricalEvalDist:
    def __init__(
            self, 
            _pdf: Callable[[NDArray[_ftype]], NDArray[_ftype]] | None = None, 
            _Cauchy: Callable[[NDArray[_ftype]], NDArray[_ftype]] | None = None, 
            _Stransform: Callable[[NDArray[_ctype]], NDArray[_ctype]] | None = None
            ) -> None:
        
        self._pdf = _pdf
        self._Cauchy = _Cauchy
        self._Stransform = _Stransform

        if _pdf is None and _Cauchy is None and _Stransform is None:
            raise ValueError("Please provide any of the PDF, Cauchy transform, or S transform of the distribution")
        
    def __repr__(self) -> str:
        provided = [
            name
            for name, fn in (
                ("pdf", self._pdf),
                ("Cauchy", self._Cauchy),
                ("Stransform", self._Stransform),
            )
            if fn is not None
        ]
        return f"EmpiricalEvalDist(from={'+'.join(provided)})"

    def support(self):
        pass

    def pdf(self, x: NDArray[_ftype], eps: _ftype = 1e-7) -> NDArray[_ftype]:
        if self._pdf is not None:
            return self._pdf(x)
        
        elif self._Cauchy is not None:
            return (-1.0 / np.pi) * np.imag(self._Cauchy(x + 1j * eps))
        
        else:
            return (-1.0 / np.pi) * np.imag(self._Cauchy_from_S(x + 1j * eps))

    def Cauchy(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        if self._Cauchy is not None:
            return self._Cauchy(z)
        
        elif self._pdf is not None:
            t = np.linspace(0.01, np.pi / 2, 10000)[np.newaxis, :]
            x = np.tan(t)
            dx_dt = 1.0 / np.cos(t) ** 2

            z = z[:, np.newaxis]
            integrand = 1 / (z - x) * self._pdf(x) * dx_dt 

            return np.trapezoid(integrand, t, axis=1)
        
        else:
            return self._Cauchy_from_S(z)

    def Rtransform(self):
        pass

    def Stransform(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        if self._Stransform is not None:
            return self._Stransform(z)
        
        else:
            return self._S_from_Cauchy(z)
        
    def _S_from_Cauchy(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        
        def target_func(u: NDArray[_ctype], z: NDArray[_ctype]) -> NDArray[_ctype]:
            return self.Cauchy(1/u) / u - 1.0 - z
        
        def initial_guess_func(z: NDArray[_ctype]) -> NDArray[_ctype]:
            return 1.0 / z
        
        z_flat = np.atleast_1d(z).ravel()
        
        u = self._finding_function_inverse(z_flat, initial_guess_func, target_func)

        u_ld = u.astype(np.clongdouble)
        z_ld = z_flat.astype(np.clongdouble)
        out_ld = u_ld * (z_ld + 1.0) / z_ld
        out = out_ld.astype(_ctype)
        return out.reshape(z.shape)

    def _Cauchy_from_S(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        
        def target_func(u: NDArray[_ctype], z: NDArray[_ctype]) -> NDArray[_ctype]:
            return self.Stransform(u) * u / (u + 1.0) - 1.0 / z
        
        def initial_guess_func(z: NDArray[_ctype]) -> NDArray[_ctype]:
            return 1.0 / z

        z_flat = np.atleast_1d(z).ravel()
        
        u = self._finding_function_inverse(z_flat, initial_guess_func, target_func)

        u_ld = u.astype(np.clongdouble)
        z_ld = z_flat.astype(np.clongdouble)
        out_ld = (1.0 + u_ld) / z_ld
        out = out_ld.astype(_ctype)
        return out.reshape(z.shape)

    def _finding_function_inverse(self, z: NDArray[_ctype], initial_guess_func: Callable[[NDArray[_ctype]], NDArray[_ctype]], target_func: Callable[[NDArray[_ctype], NDArray[_ctype]], NDArray[_ctype]], tol: _ftype = 1e-14, max_iter: int = 100) -> NDArray[_ctype]:
        n = len(z)
        z_real = np.real(z)

        eps_target = np.fmax(np.abs(np.imag(z)), 1e-7)
        eps_start = 10.0
        ratio = 10 ** 0.5
        eps_steps = int(np.ceil(np.log10(eps_start / eps_target.min()) / np.log10(ratio))) + 1
        eps_sequence = np.geomspace(eps_start, eps_target, num=eps_steps)
        u = initial_guess_func(z_real + 1j * eps_sequence[0])

        for i in range(eps_steps):
            z_i = z_real + 1j * eps_sequence[i]

            converged = np.zeros(n, dtype=bool)

            for _ in range(max_iter):
                active = ~converged
                if not np.any(active):
                    break

                u_a, z_a = u[active], z_i[active]

                u_new, f, f_new = self._newton_step_batch(u_a, z_a, target_func)
                u_new = self._backtrack_batch(u_a, u_new, f, f_new, z_a, target_func)

                delta = np.abs(u_new - u_a)
                newly_converged = (delta < tol * np.maximum(1.0, np.abs(u_a)))
                converged[active] = newly_converged
                u[active] = u_new

        return u

    def _newton_step_batch(self, u: NDArray[_ctype], z: NDArray[_ctype], target_func: Callable[[NDArray[_ctype], NDArray[_ctype]], NDArray[_ctype]]):
        f = target_func(u, z)
        u1 = u * 1.01
        f1 = target_func(u1, z)
        df = (f1 - f + 1e-15) / (0.01 * u)

        u_new = u - f / df 
        u_new = np.where(np.isfinite(u_new), u_new, u)
        u_new = np.where(np.imag(u_new)>0, u_new.conj(), u_new)

        f_new = target_func(u_new, z)

        return u_new, f, f_new
    
    def _backtrack_batch(self, u: NDArray[_ctype], u_new: NDArray[_ctype], f: NDArray[_ctype], f_new: NDArray[_ctype], z: NDArray[_ctype], target_func: Callable[[NDArray[_ctype], NDArray[_ctype]], NDArray[_ctype]]):
        alpha = np.ones(len(u), dtype=_ftype)
        thresh = np.abs(f) * 1.1

        needs_bt = np.abs(f_new) > thresh
        iters = 0

        while np.any(needs_bt) and np.any(alpha > 1e-8) and iters < 20:
            alpha[needs_bt] *= 0.5
            u_try = u + alpha * (u_new - u)
            u_try = np.where(np.imag(u_try)>0, u_try.conj(), u_try)

            f_try = target_func(u_try, z)
            improved = np.abs(f_try) < np.abs(f_new)

            u_new = np.where(improved, u_try, u_new)
            f_new = np.where(improved, f_try, f_new)

            needs_bt = (np.abs(f_new) > thresh)
            iters += 1

        return u_new


class MarchenkoPastur(EmpiricalEvalDist):
    def __init__(self, lam: _ftype, sig: _ftype = 1.0):
        if lam <= 0: 
            raise ValueError(f"lambda must be positive, got {lam}")
        if sig <= 0:
            raise ValueError(f"sigma must be positive, got {sig}")
        
        self.lam = lam
        self.sig = sig
        self._sqrt_lam = np.sqrt(self.lam)
        self.lam_plus = self.sig ** 2 * (1.0 + self._sqrt_lam) ** 2
        self.lam_minus = self.sig ** 2 * (1.0 - self._sqrt_lam) ** 2
        self.point_mass_at_zero = max(0.0, 1.0 - 1.0 / self.lam)
    
    def __repr__(self) -> str:
        return (
            f"MarchenkoPastur("
            f"lambda={self.lam}, sigma={self.sig}, "
            f"support=[{self.lam_minus:.4f}, {self.lam_plus:.4f}], "
            f"{'point_mass_at_0='+f'{self.point_mass_at_zero:.4f}' if self.point_mass_at_zero > 0 else ''})"
        )
    
    def support(self) -> tuple[_ftype, _ftype]:
        return (self.lam_minus, self.lam_plus)
    
    def pdf(self, x: NDArray[_ftype]) -> NDArray[_ftype]:
        shape = x.shape
        x = np.atleast_1d(x)

        out = np.zeros_like(x)

        mask = (x > self.lam_minus) & (x <= self.lam_plus) & (x > 0)
        xm = x[mask]
        numerator = np.sqrt((self.lam_plus - xm) * (xm - self.lam_minus))
        denominator = 2.0 * np.pi * self.sig ** 2 * self.lam * xm
        out[mask] = numerator / denominator

        # mask = x == 0
        # xm = x[mask]
        # out[mask] = self.point_mass_at_zero

        return out.reshape(shape)
    
    def Cauchy(self, z: NDArray[_ctype]) -> NDArray[_ctype]: 
        return (self.sig ** 2 * (1.0 - self.lam) - z - np.sqrt((z - self.sig ** 2 * (1.0 + self.lam)) ** 2 - 4.0 * self.lam * self.sig ** 4)) / (2.0 * self.lam * z * self.sig ** 2)
    
    def Rtransform(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        return self.sig ** 2 / (1.0 - self.sig ** 2 * self.lam * z)

    def Stransform(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        return 1.0 / (1.0 + self.lam * z) / self.sig ** 2

def free_multiplicative_convolution(dist1: EmpiricalEvalDist, dist2: EmpiricalEvalDist) -> EmpiricalEvalDist:

    def Stransform_prod(z: NDArray[_ctype]) -> NDArray[_ctype]:
        return dist1.Stransform(z) * dist2.Stransform(z)
    
    return EmpiricalEvalDist(_Stransform=Stransform_prod)

def classical_multiplicative_convolution(dist1: EmpiricalEvalDist, dist2: EmpiricalEvalDist) -> EmpiricalEvalDist:
    
    def pdf_prod(x: NDArray[_ftype]) -> NDArray[_ftype]:
        # Computing classical multiplicative convolution using fast fourier transform 

        n = 10000
        conv_len = 2 * n - 1

        t_grid = np.geomspace(1e-6, 5, n)
        t_log = np.log(t_grid)
        dt_log = t_log[1] - t_log[0]

        f = dist1.pdf(t_grid) * t_grid
        g = dist2.pdf(t_grid) * t_grid

        conv = np.fft.irfft(np.fft.rfft(f, n=conv_len) * np.fft.rfft(g, n=conv_len), n=conv_len) * dt_log

        t_out = np.exp(np.linspace(t_log[0] + t_log[0], t_log[-1] + t_log[-1], conv_len))
        conv_out = conv / t_out

        return np.interp(x, t_out, conv_out, left=0.0, right=0.0)

    return EmpiricalEvalDist(_pdf=pdf_prod)

class _Module:
    dist: "EmpiricalEvalDist"
    first_k: int
    last_k: int
    first_in: list
    last_out: list

    def __mul__(self, other):          # series   -> free convolution
        if not isinstance(other, _Module):
            return NotImplemented
        return Sequential(self, other)

    def __add__(self, other):          # parallel -> classical convolution
        if not isinstance(other, _Module):
            return NotImplemented
        return Parallel(self, other)

    def __pow__(self, n):              # repeated series
        m = self
        for _ in range(int(n) - 1):
            m = Sequential(m, self)
        return m

    def __rmul__(self, n):            # int * module -> repeated parallel
        m = self
        for _ in range(int(n) - 1):
            m = Parallel(m, self)
        return m

class Neuron(_Module):
    def __init__(self, d_in, d_out, sig=1.0):
        self.d_in, self.d_out, self.sig = int(d_in), int(d_out), sig
        self.dist = MarchenkoPastur(self.d_out / self.d_in, sig)
        self.first_k = self.last_k = 1
        self.first_in = [self.d_in]
        self.last_out = [self.d_out]


def _check_series(a: _Module, b: _Module):
    """Check that 
    (1) last_out vs first_k : a's common output dim must split equally into b's
        first-layer neurons.
    (2) last_k  vs first_in : b's common input dim must split equally into a's
        last-layer neurons.
    """
    if len(set(a.last_out)) != 1:
        raise ValueError(f"last-layer output dims must be equal, got {a.last_out}")
    if len(set(b.first_in)) != 1:
        raise ValueError(f"first-layer input dims must be equal, got {b.first_in}")
    d_out, d_in = a.last_out[0], b.first_in[0]
    if d_out % b.first_k != 0:                        # (1) last_out vs first_k
        raise ValueError(f"output dim {d_out} not divisible by {b.first_k} first-layer neurons")
    if d_in % a.last_k != 0:                          # (2) last_k vs first_in
        raise ValueError(f"input dim {d_in} not divisible by {a.last_k} last-layer neurons")

class Sequential(_Module):
    def __init__(self, *items, sig=1.0):
        if all(isinstance(x, int) for x in items):
            if len(items) < 2:
                raise ValueError("need at least two connecting dims")
            mods = [Neuron(items[i], items[i + 1], sig) for i in range(len(items) - 1)]
        elif all(isinstance(x, _Module) for x in items):
            mods = list(items)
        else:
            raise TypeError("Sequential args must be all ints or all modules")

        dist = mods[0].dist
        for a, b in zip(mods, mods[1:]):
            _check_series(a, b)
            dist = free_multiplicative_convolution(dist, b.dist)

        self.dist = dist
        self.first_k, self.first_in = mods[0].first_k, mods[0].first_in
        self.last_k, self.last_out = mods[-1].last_k, mods[-1].last_out

class Parallel(_Module):
    def __init__(self, *modules):
        if len(modules) < 2 or not all(isinstance(m, _Module) for m in modules):
            raise TypeError("Parallel needs >= 2 modules (Neuron/Sequential/Parallel)")

        dist = modules[0].dist
        for m in modules[1:]:
            dist = classical_multiplicative_convolution(dist, m.dist)

        self.dist = dist
        self.first_k = sum(m.first_k for m in modules)
        self.last_k = sum(m.last_k for m in modules)
        self.first_in = [x for m in modules for x in m.first_in]
        self.last_out = [x for m in modules for x in m.last_out]

def infer(network, d):
    """Concrete _Module for a series-parallel `network`, given the integer `d`
    that is the dimension of the space connecting each pair of neurons.

    `network` is either a _Module (returned as-is) or a callable u -> _Module
    written with the canonical * / + operators, e.g. ``lambda u: (u * u) + u``.
    Every connecting space is assigned dimension `d`, so each neuron is d -> d;
    the series checks then constrain which `d` are valid."""
    if isinstance(network, _Module):
        return network
    return network(Neuron(d, d))