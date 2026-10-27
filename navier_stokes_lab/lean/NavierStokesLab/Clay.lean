/-
Laboratorio Navier–Stokes (Lean 4).

Clay Millennium statement for Navier–Stokes, recorded as a proposition.

This is *not* a proof. It pins down the claim the laboratory is exploring:
global smooth solutions of 3D incompressible Navier–Stokes on ℝ³ or 𝕋³
for smooth, divergence-free, finite-energy initial data.

Reference: Charles Fefferman, “Existence and smoothness of the
Navier–Stokes equation”, Clay Mathematics Institute.
-/

namespace NavierStokesLab

/-- Smooth, compactly supported (or Schwartz) initial velocity. -/
structure SmoothInitial (n : Nat) where
  dummy : Unit := ()

/-- Finite kinetic energy. -/
class FiniteEnergy {n : Nat} (u : SmoothInitial n) : Prop

/-- Divergence-free constraint ∇ · u₀ = 0. -/
class DivergenceFree {n : Nat} (u : SmoothInitial n) : Prop

/-- A global-in-time smooth solution of incompressible Navier–Stokes. -/
structure GlobalSmoothSolution (n : Nat) where
  initial : SmoothInitial n
  dummy : Unit := ()

/-- Official Clay conjecture (existence and smoothness), dimension 3.

    ∀ u₀ smooth, divergence-free, finite energy,
      there exists a unique global smooth solution.

Uniqueness is left implicit in `Nonempty` for this skeleton; a later
Mathlib layer should split existence and uniqueness. -/
def ClayExistenceSmoothness : Prop :=
  ∀ u0 : SmoothInitial 3,
    FiniteEnergy u0 →
    DivergenceFree u0 →
    Nonempty (GlobalSmoothSolution 3)

/-- The 2D theory is classical (Ladyzhenskaya, 1960s): we record it as
an axiom of this laboratory, not as a theorem of analysis. -/
axiom twoDimensionalGlobalRegularity :
    ∀ u0 : SmoothInitial 2,
      FiniteEnergy u0 →
      DivergenceFree u0 →
      Nonempty (GlobalSmoothSolution 2)

/-- Computational proxy used by the Python solver: Taylor–Green on 𝕋²
is globally smooth for every ν ≥ 0. -/
axiom taylorGreenIsGlobalSmooth : True

end NavierStokesLab
