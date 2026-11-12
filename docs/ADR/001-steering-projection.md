# 001: Orthogonal Projection for Model Steering

## Status
Accepted

## Context
When applying steering vectors to the model activations (specifically in the residual stream), simple addition (`hidden_states + vector`) can alter the magnitude of the residual stream, potentially causing training instability or quality degradation at high strengths.

## Decision
Use orthogonal projection to apply steering vectors, as implemented in the `ds4` research stack.
Formula: `y = y - scale * (y @ v) * v`, where `v` is the normalized steering vector.

## Consequences
- Preserves the norm of the residual stream.
- Steering strength is more predictable and robust at higher values.
- Requires pre-normalized steering vectors (`norm == 1`).
