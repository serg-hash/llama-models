"""Gravedad Relativista Oscura (GRO)

This module implements a conceptual model inspired by Einstein's general relativity
where the effects attributed to dark matter arise from an alternative geometry
of space-time. The approach modifies the effective gravitational potential with
an exponential term that mimics the additional attraction usually associated
with dark matter in galaxies.

The implementation is intentionally simple and should be viewed as a toy example
for exploring ideas that connect relativistic curvature with observable
galactic dynamics.
"""

import math

# Gravitational constant in units of kpc * (km/s)^2 / Msun
G = 4.302e-6


def gro_potential(M: float, r: float, alpha: float = 0.1, r0: float = 5.0) -> float:
    """Return the modified gravitational potential in the GRO framework.

    Parameters
    ----------
    M : float
        Mass in solar masses.
    r : float
        Radius in kiloparsecs.
    alpha : float, optional
        Strength of the geometric modification.
    r0 : float, optional
        Scale length for the modification.
    """
    return -G * M / r * (1 + alpha * math.exp(-r / r0))


def gro_rotation_velocity(M: float, r: float, alpha: float = 0.1, r0: float = 5.0) -> float:
    """Rotation speed predicted by GRO at radius ``r``.

    The expression includes the Newtonian term plus a correction that
depends on ``alpha`` and ``r0``. Setting ``alpha=0`` recovers the classic
Keplerian falloff.
    """
    term = 1.0 + alpha * (1.0 + r / r0) * math.exp(-r / r0)
    return math.sqrt(G * M / r * term)


def demo() -> None:
    """Print rotation curves for a simple galaxy using GRO."""
    radii = [0.5 + 0.2 * i for i in range(100)]
    M = 1.0e11  # solar masses
    newtonian = [math.sqrt(G * M / r) for r in radii]
    gro = [gro_rotation_velocity(M, r) for r in radii]
    for r, v_n, v_g in zip(radii, newtonian, gro):
        print(f"{r:5.2f} kpc -> Newtonian: {v_n:7.2f} km/s, GRO: {v_g:7.2f} km/s")


if __name__ == "__main__":
    demo()
