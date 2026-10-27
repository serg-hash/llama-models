# Lean 4 — NavierStokesLab

```bash
lake build
```

| Archivo | Rol |
| --- | --- |
| `NavierStokesLab/Equations.lean` | Esqueleto de NS incompresible y de la forma en vorticidad 2D |
| `NavierStokesLab/Leray.lean` | Proyector discreto: `k · L = 0`, idempotencia, modos fijos |
| `NavierStokesLab/Energy.lean` | Identidad de energía: `d/dt (½ ‖u‖²) + ν ‖∇u‖² = 0` (`grind`, sin `sorry`) |
| `NavierStokesLab/Clay.lean` | Enunciado Clay (`Prop`) y regularidad 2D como axioma |

`ClayExistenceSmoothness` no debe convertirse en `theorem` sin una prueba
analítica real. Ver `../prompts/siguiente_lema.md`.
