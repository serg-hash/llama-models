# Auto-improvement pipeline

This module provides a small example of an automatic improvement loop for
searching zeros of the Riemann zeta function using Hardy's `Z(t)`.

## Running

```bash
python -m riemann.auto_improve.runner
```

Results are stored in `runs/latest/` with the configuration in `codex.yaml`
and a `report.md` summarising each iteration.

## Extending

Operators for mutation or selection can be added in `mutator.py` and
`selector.py`. The evaluator exposes a simple interface for testing new
strategies.
