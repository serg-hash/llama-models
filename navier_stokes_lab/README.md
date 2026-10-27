# Laboratorio Navier–Stokes

Laboratorio matemático **2D + Lean 4** para las ecuaciones de Navier–Stokes
incompresibles, dentro de este fork. Es el arranque público del trabajo que
vive en el repo privado `NAVIER-STOKES-CLOUD`.

No resuelve el problema del milenio de Clay. El objetivo es tener:

1. Un **solver espectral 2D** con una solución exacta (Taylor–Green) contra la
   que medir error.
2. Un **esqueleto Lean 4** con el enunciado de Clay, las identidades
   algebraicas del proyector de Leray y el balance de energía
   `d/dt (½ ‖u‖²) + ν ‖∇u‖² = 0` (el mismo objeto que usa el solver).
3. Un **prompt** para continuar las pruebas formales con Codex / Cursor.

## Python

```bash
python -m navier_stokes_lab --n-grid 32 --nu 0.05 --dt 0.005 --steps 40
python -m navier_stokes_lab.test_lab
```

Equivalente en la raíz del repo:

```bash
python navier_stokes_explorer.py --steps 40
```

Dependencia: `numpy` (ver `requirements.txt`).

El esquema es la formulación en vorticidad en el toro `[0, 2π]²`:

```
∂t ω + u · ∇ω = ν Δω
Δψ = −ω,   u = (∂y ψ, −∂x ψ)
```

Taylor–Green anula el término no lineal. La energía cinética media es
`(1/4) exp(-4 ν t)`.

## Lean 4

```bash
cd navier_stokes_lab/lean
lake build
```

Toolchain: `leanprover/lean4:v4.22.0` (sin Mathlib). Los lemas de
`NavierStokesLab/Leray.lean` y `NavierStokesLab/Energy.lean` son
identidades en `Int` y deben cerrar con `grind`. `Clay.lean` registra
la conjetura como `Prop`, no como teorema.

Siguiente paso formal: `prompts/siguiente_lema.md`.

## Qué queda fuera (a propósito)

- Existencia y suavidad en 3D (Clay).
- Mathlib / espacios de Sobolev.
- Solver 3D y geometrías no periódicas.
