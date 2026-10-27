/-
Laboratorio Navier–Stokes (Lean 4).

Discrete Leray projector on Fourier modes with integer coordinates.

For a wavevector k = (kx, ky) the (unnormalized) projector is

    L(k, u) = |k|² u − k (k · u)

This is the numerator of P̂(k) u = u − k (k · u)/|k|².
Working with the numerator keeps every identity inside `Int`, so the
lemmas compile without Mathlib. The same identities are tested in
`navier_stokes_lab/test_lab.py`.
-/

namespace NavierStokesLab

/-- |k|² for a 2D integer wavevector. -/
def ksq (kx ky : Int) : Int := kx * kx + ky * ky

/-- k · u. -/
def kdot (kx ky ux uy : Int) : Int := kx * ux + ky * uy

/-- Numerator of the Leray projector: |k|² u − k (k · u). -/
def lerayNum (kx ky ux uy : Int) : Int × Int :=
  (ksq kx ky * ux - kx * kdot kx ky ux uy,
   ksq kx ky * uy - ky * kdot kx ky ux uy)

/-- k · L(k, u) = 0. The projected mode is divergence-free. -/
theorem leray_div_free (kx ky ux uy : Int) :
    kdot kx ky (lerayNum kx ky ux uy).1 (lerayNum kx ky ux uy).2 = 0 := by
  unfold lerayNum kdot ksq
  grind

/-- L(k, L(k, u)) = |k|² L(k, u). Idempotence of P up to the |k|² scale. -/
theorem leray_idempotent_numerator (kx ky ux uy : Int) :
    lerayNum kx ky (lerayNum kx ky ux uy).1 (lerayNum kx ky ux uy).2 =
      (ksq kx ky * (lerayNum kx ky ux uy).1,
       ksq kx ky * (lerayNum kx ky ux uy).2) := by
  unfold lerayNum ksq kdot
  grind

/-- If k · u = 0 then L(k, u) = |k|² u: divergence-free modes are fixed. -/
theorem leray_fixes_div_free (kx ky ux uy : Int)
    (h : kdot kx ky ux uy = 0) :
    lerayNum kx ky ux uy = (ksq kx ky * ux, ksq kx ky * uy) := by
  unfold lerayNum
  simp [h]

end NavierStokesLab
