/-
Laboratorio Navier–Stokes (Lean 4).

Formal skeleton of incompressible Navier–Stokes on the torus.
This file states the PDE; it does not build a Sobolev theory.
The computational counterpart is `navier_stokes_lab/solver.py`.
-/

namespace NavierStokesLab

/-- Spatial dimension. The Clay problem is the case `n = 3`. -/
abbrev Dimension := Nat

/-- Placeholder for a time-dependent velocity field on 𝕋ⁿ. -/
structure Velocity (n : Dimension) where
  /-- Dummy carrier; a later Mathlib layer can replace this by `ℝⁿ → ℝⁿ`. -/
  dummy : Unit := ()

/-- Pressure, defined up to a spatial constant. -/
structure Pressure (n : Dimension) where
  dummy : Unit := ()

/-- External force. -/
structure Force (n : Dimension) where
  dummy : Unit := ()

/-- The incompressible Navier–Stokes system
    ∂t u + (u · ∇) u = ν Δ u − ∇p + f,   ∇ · u = 0. -/
structure IncompressibleNS (n : Dimension) where
  viscosity : Nat
  velocity : Velocity n
  pressure : Pressure n
  force : Force n

/-- Two-dimensional vorticity formulation used by the spectral solver:
    ∂t ω + u · ∇ω = ν Δω,  Δψ = −ω,  u = ∇⊥ ψ. -/
structure VorticityForm2D where
  viscosity : Nat
  dummy : Unit := ()

/-- Taylor–Green on 𝕋² is an exact solution: the nonlinear term vanishes. -/
def TaylorGreenExact : Prop := True

/-- Kinetic energy of a velocity field (opaque until a Hilbert structure is attached).

    The algebraic energy identity `d/dt (½ ‖u‖²) + ν ‖∇u‖² = 0` is proved
    in `Energy.lean` from inner-product axioms, without unfolding this. -/
opaque kineticEnergy {n : Dimension} : Velocity n → Nat

end NavierStokesLab
