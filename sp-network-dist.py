import numpy as np
from numpy.typing import NDArray
from typing import Callable

_ftype = np.float64
_ctype = np.complex128

class EmpricalEvalDist:
    def __init__(
            self, 
            _pdf: Callable[[NDArray[_ftype]], NDArray[_ftype]] | None = None, 
            _Cauchy: Callable[[NDArray[_ftype]], NDArray[_ftype]] | None = None, 
            _Stransform: Callable[[NDArray[_ctype]], NDArray[_ctype]] | None = None
            ) -> None:
        if any([_pdf is not None, _Cauchy is not None, _Stransform is not None]):
            self._pdf = _pdf
            self._Cauchy = _Cauchy
            self._Stransform = _Stransform

        else: 
            raise ValueError("Please provide any of the PDF, Cauchy transform, or S transform of the distribution")
        
    def __repr__(self) -> str:
        pass

    def support(self):
        pass

    def pdf(self, x: NDArray[_ftype], eps: _ftype = 1e-7) -> NDArray[_ftype]:
        x = np.asarray(x, dtype=_ftype)

        if self._pdf is not None:
            return self._pdf(x)
        
        elif self._Cauchy is not None:
            return (-1.0 / np.pi) * np.imag(self._Cauchy(x + 1j * eps))
        
        else:
            return (-1.0 / np.pi) * np.imag(self._Cauchy_from_S(x + 1j * eps))

    def Cauchy(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        z = np.asarray(z, dtype=_ctype)

        if self._Cauchy is not None:
            return self._Cauchy(z)
        
        elif self._pdf is not None:
            t = np.linspace(-10, 10, 10000)[np.newaxis, :]
            x = np.sinh(t)
            dx_dt = np.cosh(t)

            z = z[:, np.newaxis]
            integrand = 1 / (z - x) * self._pdf(x) * dx_dt 

            return np.trapezoid(integrand, t, axis=1)
        
        else:
            return self._Cauchy_from_S(z)

    def Rtransform(self):
        pass

    def Stransform(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        z = np.asarray(z, dtype=_ctype)

        if self._Stransform is not None:
            return self._Stransform(z)
        
        else:
            return self._S_from_Cauchy(z)
        
    def _S_from_Cauchy(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        
        def target_func(u: NDArray[_ctype], z: NDArray[_ctype]) -> NDArray[_ctype]:
            return self.Cauchy(1/u) / u - 1.0 - z
        
        def initial_guess_func(z: NDArray[_ctype]) -> NDArray[_ctype]:
            return 1.0 / z
            
        z = np.asarray(z, dtype=_ctype)
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

        z = np.asarray(z, dtype=_ctype)
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

        eps_target = np.imag(z)
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


class MarchenkoPastur:
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
        x = np.asarray(x, dtype=_ftype)
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
        z = np.asarray(z, dtype=_ctype)
        return self.sig ** 2 / (1.0 - self.sig ** 2 * self.lam * z)

    def Stransform(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        z = np.asarray(z, dtype=_ctype)
        return 1.0 / (1.0 + self.lam * z) / self.sig ** 2

def free_multiplicative_convolution(dist1: EmpricalEvalDist, dist2: EmpricalEvalDist) -> EmpricalEvalDist:

    def Stransform_prod(z: NDArray[_ctype]) -> NDArray[_ctype]:
        return dist1.Stransform(z) * dist2.Stransform(z)
    
    return EmpricalEvalDist(_Stransform=Stransform_prod)

def classical_multiplicative_convolution(dist1: EmpricalEvalDist, dist2: EmpricalEvalDist) -> EmpricalEvalDist:
    
    def pdf_prod(x: NDArray[_ftype]) -> NDArray[_ftype]:
        t = np.linspace(-10, 10, 10000)[np.newaxis, :]
        u = np.sinh(t)
        du_dt = np.cosh(t)

        x = x[:, np.newaxis]
        integrand = dist1.pdf(x / u) * dist2.pdf(u) / u * du_dt 
        return np.trapezoid(integrand, t, axis=1)

    return EmpricalEvalDist(_pdf=pdf_prod)
