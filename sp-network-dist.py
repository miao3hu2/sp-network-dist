import numpy as np
from numpy.typing import NDArray
from typing import Callable

_ftype = np.float64
_ctype = np.complex128

class EmpricalEvalDist:
    def __init__(self, _pdf: Callable[[NDArray[_ftype]], NDArray[_ftype]] | None = None, _Stransform: Callable[[NDArray[_ctype]], NDArray[_ctype]] | None = None, _dStransform: Callable[[NDArray[_ctype]], NDArray[_ctype]] | None = None) -> None:
        if _pdf is not None:
            self._pdf = _pdf
            self._Stransform = None
            self._dStransform = None
        elif _Stransform is not None:
            self._pdf = None
            self._Stransform = _Stransform
            self._dStransform = _dStransform
        else:
            raise ValueError("Provide `probility density function` or `S transofrm` together with its first derivative.")
        
    def __repr__(self) -> str:
        pass

    def support(self):
        pass

    def pdf(self, x: NDArray[_ftype], eps: _ftype = 1e-7) -> NDArray[_ftype]:
        x = np.asarray(x, dtype=_ftype)
        if self._pdf is not None:
            return self._pdf(x)
        else:
            return (-1.0 / np.pi) * np.imag(self.Caychytransform(x + 1j * eps))

    def Caychytransform(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        z = np.asarray(z, dtype=_ctype)
        if self._pdf is not None:
            pass
        else:
            return self._Cauchy_from_S(z)

    def Rtransform(self):
        pass

    def Stransform(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        z = np.asarray(z, dtype=_ctype)
        if self._pdf is not None:
            pass
        else:
            return self._Stransform(z)
        
    def dStransform(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        z = np.asarray(z, dtype=_ctype)
        if self._pdf is not None:
            pass
        else:
            return self._dStransform(z)

    def _Cauchy_from_S(
            self,
            z: NDArray[_ctype],
            tol: _ftype = 1e-14,
            tol_res: _ftype = 1e-13,
            max_iter: int = 100
            ) -> NDArray[_ctype]:
        z = np.asarray(z, dtype=_ctype)
        z_flat = np.atleast_1d(z).ravel()
        n = len(z_flat)
        z_real = np.real(z_flat)

        eps_target = np.imag(z_flat)
        eps_start = 10.0
        ratio = 10 ** 0.5
        eps_steps = int(np.ceil(np.log10(eps_start / eps_target.min()) / np.log10(ratio))) + 1
        eps_sequence = np.geomspace(eps_start, eps_target, num=eps_steps)
        u = 1.0 / (z_real + 1j * eps_sequence[0])

        for i in range(eps_steps):
            z_i = z_real + 1j * eps_sequence[i]

            converged = np.zeros(n, dtype=bool)

            for _ in range(max_iter):
                active = ~converged
                if not np.any(active):
                    break

                u_a, z_a = u[active], z_i[active]

                u_new, f, f_new = self._newton_step_batch(u_a, z_a)
                u_new = self._backtrack_batch(u_a, u_new, f, f_new, z_a)

                delta = np.abs(u_new - u_a)
                f_new_res = np.abs(u_new * (self._Stransform(u_new) * u_new / (1.0 + u_new) - 1.0 / z_a))
                newly_converged = ((delta < tol * np.maximum(1.0, np.abs(u_a))) & (f_new_res < tol_res))
                converged[active] = newly_converged
                u[active] = u_new

        u_ld = u.astype(np.clongdouble)
        z_ld = z_flat.astype(np.clongdouble)
        out_ld = (1.0 + u_ld) / z_ld
        out = out_ld.astype(_ctype)
        return out.reshape(z.shape)
    
    def _newton_step_batch(self, u: NDArray[_ctype], z: NDArray[_ctype]):
        S_u = self._Stransform(u)
        dS_u = self._dStransform(u)

        u_plus_1 = 1.0 + u
        f = u * (S_u * u / u_plus_1 - 1.0 / z)
        df = u * (dS_u * u / u_plus_1 + S_u / u_plus_1 ** 2) + f

        safe = np.abs(df) >= 1e-14
        u1 = np.where(safe, u, u * 1.01)
        S_u1 = np.where(safe, S_u, self._Stransform(u1))
        f1 = np.where(safe, f, u1 * (S_u1 * u1 / (1.0 + u1) - 1.0 / z))
        df = np.where(safe, df, (f1 - f + 1e-15) / (0.01 * u))

        u_new = u - f/df
        u_new = np.where(np.imag(u_new)>0, u_new.conj(), u_new)

        f_new = u_new * (self._Stransform(u_new) * u_new / (1.0 + u_new) - 1.0 / z)
        return u_new, f, f_new
    
    def _backtrack_batch(self, u: NDArray[_ctype], u_new: NDArray[_ctype], f: NDArray[_ctype], f_new: NDArray[_ctype], z: NDArray[_ctype]):
        alpha = np.ones(len(u), dtype=_ftype)
        thresh = np.abs(f) * 1.1

        needs_bt = np.abs(f_new) > thresh
        iters = 0

        while np.any(needs_bt) and np.any(alpha > 1e-8) and iters < 20:
            alpha[needs_bt] *= 0.5
            u_try = u + alpha * (u_new - u)
            u_try = np.where(np.imag(u_try)>0, u_try.conj(), u_try)

            f_try = u_try * (self._Stransform(u_try) * u_try / (1.0 + u_try) - 1.0 / z)
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
    
    def Caychytransform(self, z: NDArray[_ctype]) -> NDArray[_ctype]: 
        pass

    # def Rtransform(self, z: sp.Symbol) -> sp.Symbol:
    #     return self.sig ** 2 / (1 - self.sig ** 2 * self.lam * z)
    
    def Rtransform(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        z = np.asarray(z, dtype=_ctype)
        return self.sig ** 2 / (1.0 - self.sig ** 2 * self.lam * z)

    # def Stransform(self, z: sp.Symbol) -> sp.Symbol:
    #     return 1 / (1 + self.lam * z) / self.sig ** 2

    def Stransform(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        z = np.asarray(z, dtype=_ctype)
        return 1.0 / (1.0 + self.lam * z) / self.sig ** 2
    
    def dStransform(self, z: NDArray[_ctype]) -> NDArray:
        z = np.asarray(z, dtype=_ctype)
        return -self.lam / (self.sig * (1.0 + self.lam * z)) ** 2

def free_multiplicative_convolution(dist1: EmpricalEvalDist, dist2: EmpricalEvalDist) -> EmpricalEvalDist:

    def Stransform_prod(z: NDArray[_ctype]) -> NDArray[_ctype]:
        return dist1.Stransform(z) * dist2.Stransform(z)
    
    def dStransform_prod(z: NDArray[_ctype]) -> NDArray[_ctype]:
        return dist1.Stransform(z) * dist2.dStransform(z) + dist1.dStransform(z) * dist2.Stransform(z)
    
    return EmpricalEvalDist(_Stransform=Stransform_prod, _dStransform=dStransform_prod)

def classical_multiplicative_convolution(dist1: EmpricalEvalDist, dist2: EmpricalEvalDist) -> EmpricalEvalDist:
    
    def pdf_prod(x: NDArray[_ftype]) -> NDArray[_ftype]:
        theta = np.linspace(-np.pi/2 + 1e-5, np.pi/2 - 1e-5, 10000)[np.newaxis, :]
        u = np.tan(theta)
        du_dtheta = 1 / np.cos(theta)**2

        x = x[:, np.newaxis]
        integrand = dist1.pdf(x / u) * dist2.pdf(u) / u * du_dtheta 
        return np.trapezoid(integrand, theta, axis=1)
    
    return EmpricalEvalDist(_pdf=pdf_prod)
