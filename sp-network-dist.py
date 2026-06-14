import numpy as np
from numpy.typing import NDArray
from typing import Callable

_ftype = np.float64
_ctype = np.complex128

class EmpricalEvalDist:
    def __init__(self, _pdf: Callable[[NDArray[_ftype]], NDArray[_ftype]] | None = None, _Stransform: Callable[[NDArray[_ctype]], NDArray[_ctype]] | None = None) -> None:
        if _pdf is not None:
            self._pdf = _pdf
            self._Stransform = None
        elif _Stransform is not None:
            self._pdf = None
            self._Stransform = _Stransform
        else:
            raise ValueError("Provide `probility density function` or `S transofrm`.")
        
    def __repr__(self) -> str:
        pass

    def support(self):
        pass

    def pdf(self, x: NDArray[_ftype], eps: _ftype = 1e-12) -> NDArray[_ftype]:
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

    def _Cauchy_from_S(self, z: NDArray[_ctype]) -> NDArray[_ctype]:
        z = np.asarray(z, dtype=_ctype)
        z_flat = np.atleast_1d(z).ravel()
        out = np.empty_like(z_flat)
        for i, zi in enumerate(z_flat):
            u = self._find_u_for_z(zi)
            out[i] = (1.0 + u) / zi
        return out.reshape(z.shape)
    
    def _find_u_for_z(self, z: _ctype, tol: _ftype = 1e-10, max_iter: int = 300) -> _ctype:
        u = 1.0 / z if z != 0 else 1.0 + 0j

        def Fu(u_: _ctype) -> _ctype:
            return _ctype(self._Stransform(np.asarray(u_, dtype=_ctype))) * u_ / (1.0 + u_) - 1.0 / z 
        
        h = 1e-7j
        for _ in range(max_iter):
            f = Fu(u)
            df = (Fu(u+h) - Fu(u-h)) / (2.0 * h)
            if abs(df) < 1e-30:
                break
            step = f / df
            u -= step
            if abs(step) < tol * (1.0 + abs(u)):
                break
        return u


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

def free_multiplicative_convolution(dist1: EmpricalEvalDist, dist2: EmpricalEvalDist) -> EmpricalEvalDist:

    def Stransform_prod(z: NDArray[_ctype]) -> NDArray[_ctype]:
        return dist1.Stransform(z) * dist2.Stransform(z)
    
    return EmpricalEvalDist(_Stransform=Stransform_prod)

def classical_multiplicative_convolution(dist1: EmpricalEvalDist, dist2: EmpricalEvalDist) -> EmpricalEvalDist:
    
    def pdf_prod(x: NDArray[_ftype]) -> NDArray[_ftype]:
        n = 10000
        x_min = x.min()
        x_max = x.max()
        lo = min([x_min / dist1.support()[1], dist2.support()[0]])
        lo = int(lo) + 0.01
        denom = dist1.support()[0]
        hi = x_max / denom if denom != 0 else 10
        hi = max([hi, dist2.support()[1]])
        u = np.linspace(lo, hi, n)
        print(u)
        u = u[np.newaxis, :]
        vals = dist1.pdf(x[:, np.newaxis] / u) * dist2.pdf(u) / u
        return np.trapezoid(vals, u, axis=1)
    
    
    return EmpricalEvalDist(_pdf=pdf_prod)
