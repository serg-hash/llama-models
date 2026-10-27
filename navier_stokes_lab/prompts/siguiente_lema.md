# Prompt Codex / Lean 4 — siguiente lema

Usa este texto como instrucción para el asistente (Codex, Cursor, Lean Copilot)
cuando quieras avanzar el laboratorio formal.

```
Trabajas en navier_stokes_lab/lean (Lean 4, sin Mathlib de momento).

Objetivo: acercar ClayExistenceSmoothness a un enunciado analítico, sin
pretender resolver el problema del milenio.

Siguiente lema sugerido (elige uno y demuéstralo o redúcelo a sorry mínimo):

1. Energía: para soluciones suaves de NS incompresible con f = 0,
   d/dt (½ ‖u‖²) + ν ‖∇u‖² = 0.
   Formalízalo como identidad algebraica sobre un producto interno
   opaco `inner`, con axiomas:
     inner (Δ u) u = − inner (∇u) (∇u)   (integración por partes)
     inner (∇p) u = 0                     (si ∇·u = 0)
     inner ((u·∇)u) u = 0                 (si ∇·u = 0)
   y deduce el balance de energía.

2. Extiende Leray.lean a 3D: k = (kx,ky,kz) ∈ Int³ y las mismas
   identidades k · L(k,u) = 0 y L∘L = |k|² L.

3. Conecta el solver Python: enuncia que el término no lineal de
   Taylor–Green 2D es 0 (ya se comprueba numéricamente en test_lab.py).

No marques ClayExistenceSmoothness como teorema. No uses sorry en
Leray.lean. Ejecuta `lake build` en navier_stokes_lab/lean.
```
