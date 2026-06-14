# 20260511 Cleanup Baseline and Prior/Recovery Focus v0

## Current cleanup baseline

Commit:

```text
612e10e Strengthen short-axis stress cleanup gating
```

Status:

- This version is the current baseline for mip-stage volume/artifact cleanup.
- Qualitatively, artifact removal is already good enough to stop treating cleanup strength as the main unknown.
- The useful setting is the hard-delete cleanup path around short-axis stress / volume-stress gating, especially for large non-surface volumetric Gaussians that became forced surface-like artifacts.

Important caveat:

- Hard delete can damage render quality, even when artifact removal looks right.
- The most likely failure mode is deleting contributors that were hiding view-dependent or low-frequency render error, so the surface looks cleaner but RGB fidelity drops.
- Short-axis expansion stress has not been fully tested as a downstream-quality-preserving criterion yet; it should be evaluated as a detector, not assumed to be final.

## Direction shift

For now, deprioritize further cleanup aggressiveness tuning.

Main focus moves to:

- Prior injection: improve how SR / geometry prior information is injected into the GS or SOF path after cleanup.
- Recovery: after hard cleanup, run LR-supervised recovery to restore render quality without reintroducing large volume artifacts.
- Evaluation: compare `cleanup only` against `cleanup + recovery`, and keep the deleted-artifact subset for visual inspection.

## Practical experiment split

Recommended branches of experiments:

- `cleanup only`: use commit `612e10e` behavior as the reference for artifact removal.
- `cleanup + LR recovery`: use the recovery pass added after that baseline to test whether LR supervision restores PSNR/visual quality.
- `prior injection after cleanup`: feed the cleaned/recovered mip model into the prior-injection path and measure whether prior detail can be added without reviving the old volumetric artifacts.

## Notes for future runs

- Do not over-interpret a clean deleted subset as a final win; always check rendered views and later surface extraction.
- Prefer small recovery steps first, with `images_8` LR supervision and risk-weighted opacity/scale penalties.
- If render quality cannot recover after hard delete, relax the prune cap or move some candidates from hard delete to suppression before prior injection.
