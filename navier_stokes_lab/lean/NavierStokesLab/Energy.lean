/-
Energy identity for incompressible Navier–Stokes.

For a smooth divergence-free velocity with vanishing force,

    d/dt (½ ‖u‖²) + ν ‖∇u‖² = 0

This file does *not* construct Sobolev spaces and does *not* prove the
Clay problem. It records the identity as an algebraic consequence of
three cancellations in an opaque inner product `inner`:

  ⟨Δu, u⟩     = −⟨∇u, ∇u⟩     (integration by parts on the torus)
  ⟨∇p, u⟩     = 0              (if ∇·u = 0)
  ⟨(u·∇)u, u⟩ = 0              (if ∇·u = 0; skew-symmetry)

together with the momentum equation tested against `u`. The same decay
is checked numerically for Taylor–Green in `navier_stokes_lab/test_lab.py`.
-/

namespace NavierStokesLab

/-- Integer stand-in for one inner-product pairing ⟨·,·⟩ evaluated on a
    given smooth field. Working in `Int` keeps the proof inside core Lean,
    matching `Leray.lean`. The laboratory never unfolds a Hilbert space. -/
structure EnergyBudget where
  /-- ⟨∂t u, u⟩. For a time-independent inner product this is
      d/dt (½ ‖u‖²). -/
  innerDt : Int
  /-- ⟨Δ u, u⟩ -/
  innerLap : Int
  /-- ⟨∇u, ∇u⟩  (dissipation) -/
  dissipation : Int
  /-- ⟨(u·∇)u, u⟩ -/
  innerConvect : Int
  /-- ⟨∇p, u⟩ -/
  innerPressure : Int
  /-- ⟨f, u⟩ -/
  innerForce : Int
  /-- Viscosity ν. -/
  nu : Int

/-- Momentum equation tested in the inner product against `u`:

    ⟨∂t u + (u·∇)u − ν Δu + ∇p − f, u⟩ = 0. -/
def momentumTested (e : EnergyBudget) : Prop :=
  e.innerDt + e.innerConvect - e.nu * e.innerLap + e.innerPressure - e.innerForce = 0

/-- Integration by parts on 𝕋ⁿ (no boundary term): ⟨Δu, u⟩ = −⟨∇u, ∇u⟩. -/
def integrationByParts (e : EnergyBudget) : Prop :=
  e.innerLap = -e.dissipation

/-- Pressure is L²-orthogonal to divergence-free fields. -/
def pressureOrthogonal (e : EnergyBudget) : Prop :=
  e.innerPressure = 0

/-- Skew-symmetry of the convective term on divergence-free fields. -/
def convectionSkew (e : EnergyBudget) : Prop :=
  e.innerConvect = 0

/-- Unforced equation, f = 0. -/
def forceFree (e : EnergyBudget) : Prop :=
  e.innerForce = 0

/-- Energy balance: d/dt (½ ‖u‖²) + ν ‖∇u‖² = 0. -/
def energyBalance (e : EnergyBudget) : Prop :=
  e.innerDt + e.nu * e.dissipation = 0

/-- Formal energy identity for smooth incompressible Navier–Stokes with f = 0. -/
theorem energy_identity (e : EnergyBudget)
    (hNS : momentumTested e)
    (hIBP : integrationByParts e)
    (hP : pressureOrthogonal e)
    (hC : convectionSkew e)
    (hF : forceFree e) :
    energyBalance e := by
  unfold energyBalance momentumTested integrationByParts
         pressureOrthogonal convectionSkew forceFree at *
  grind

/-- Equivalent form: ⟨∂t u, u⟩ = −ν ⟨∇u, ∇u⟩. -/
theorem energy_identity_pairing (e : EnergyBudget)
    (hNS : momentumTested e)
    (hIBP : integrationByParts e)
    (hP : pressureOrthogonal e)
    (hC : convectionSkew e)
    (hF : forceFree e) :
    e.innerDt = -e.nu * e.dissipation := by
  have h := energy_identity e hNS hIBP hP hC hF
  unfold energyBalance at h
  grind

/-- Kinetic energy is nonincreasing when ν ≥ 0 and dissipation is nonnegative. -/
theorem energy_nonincreasing (e : EnergyBudget)
    (h : energyBalance e)
    (hν : 0 ≤ e.nu)
    (hD : 0 ≤ e.dissipation) :
    e.innerDt ≤ 0 := by
  unfold energyBalance at h
  have hprod : 0 ≤ e.nu * e.dissipation := Int.mul_nonneg hν hD
  omega

end NavierStokesLab
