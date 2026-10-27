# Prompt Codex / Lean 4 — siguiente lema

Usa este texto como instrucción para el asistente (Codex, Cursor, Lean Copilot)
cuando quieras avanzar el laboratorio formal.

```
Trabajas en navier_stokes_lab/lean (Lean 4, sin Mathlib de momento).

Objetivo: acercar ClayExistenceSmoothness a un enunciado analítico, sin
pretender resolver el problema del milenio.

Hecho: `NavierStokesLab/Energy.lean` prueba el balance
`d/dt (½ ‖u‖²) + ν ‖∇u‖² = 0` como identidad algebraica sobre
`EnergyBudget` (producto interno opaco en `Int`), con los axiomas
de integración por partes, ortogonalidad de la presión y
antisimetría del término convectivo. La contraparte numérica está
en `test_lab.py` (`test_formal_energy_identity`).
`ClayExistenceSmoothness` sigue siendo un `Prop`, no un teorema.

Siguiente lema sugerido (elige uno y demuéstralo o redúcelo a sorry mínimo):

1. Extiende Leray.lean a 3D: k = (kx,ky,kz) ∈ Int³ y las mismas
   identidades k · L(k,u) = 0 y L∘L = |k|² L.

2. Conecta el solver Python: enuncia que el término no lineal de
   Taylor–Green 2D es 0 (ya se comprueba numéricamente en test_lab.py).

3. Relaciona `kineticEnergy` de `Equations.lean` con `innerDt` de
   `Energy.lean` (derivar ½⟨u,u⟩ bajo el producto interno).

No marques ClayExistenceSmoothness como teorema. No uses sorry en
Leray.lean. Ejecuta `lake build` en navier_stokes_lab/lean.
```
