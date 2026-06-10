import numpy as np
from numpy.typing import NDArray
import sympy as sp

class EmpricalEvalDist:
    def __repr__(self):
        pass

    def support(self):
        pass

    def pdf(self):
        pass

    def Caychytransform(self):
        pass

    def Rtransform(self):
        pass

    def Stransform(self):
        pass


class MarchenkoPastur:
    def __init__(self, lam: float, sig: float = 1.0):
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
            f"support=[{self.lam_minus:.4f}, {self.lam_plus:.4f}]"
            f"{', point_mass_at_0='+f'{self.point_mass_at_zero:.4f}' if self.point_mass_at_zero > 0 else ''})"
        )
    
    def support(self) -> tuple[float, float]:
        return (self.lam_minus, self.lam_plus)
    
    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
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

        return float(out[0]) if scalar else out
    
    def Caychytransform(self, z: NDArray[np.complex64] | complex) -> NDArray[np.complex64] | complex: 
        pass

    def Rtransform(self, z: sp.Symbol) -> sp.Symbol:
        return self.sig ** 2 / (1 - self.sig ** 2 * self.lam * z)
    
    # def Rtransform(self, z: NDArray[np.complex64] | complex) -> NDArray[np.complex64] | complex:
    #     if not np.all(np.imag(z) > 0):
    #         raise ValueError("Imaginary part must be positive")
        
    #     return self.sig ** 2 / (1 - self.sig ** 2 * self.lam * z)

    def Stransform(self, z: sp.Symbol) -> sp.Symbol:
        return 1 / (1 + self.lam * z) / self.sig ** 2

    # def Stransform(self, z: NDArray[np.complex64] | complex) -> NDArray[np.complex64] | complex:
    #     if not np.all(np.imag(z) > 0):
    #         raise ValueError("Imaginary part must be positive")
    
    #     return 1 / (1 + self.lam * z) / self.sig ** 2

def free_multiplicative_convolution(dist1: EmpricalEvalDist, dist2: EmpricalEvalDist) -> EmpricalEvalDist:
    final_dist = EmpricalEvalDist
    def finalStransform(z: sp.Symbol):
        return dist1.Stransform(z) * dist2.Stransform(z)
    
    final_dist.Stransform = finalStransform

    def finalRtransform(z: sp.Symbol):
        w = sp.Symbol("w")
        f_expr = z * finalStransform(z)
        return sp.solve(sp.Eq(f_expr.subs(z, w), z), w)[0]

    final_dist.Rtransform = finalRtransform

    def finalCauchytransform(z: sp.Symbol):
        w = sp.Symbol("w")
        f_expr = (finalRtransform(z) + 1) / z
        f_expr = (MarchenkoPastur(1).Rtransform(z) + 1) / z
        return sp.solve(sp.Eq(f_expr.subs(z, w), z), w)[0]
    
    final_dist.Caychytransform = finalCauchytransform
    
    return final_dist
