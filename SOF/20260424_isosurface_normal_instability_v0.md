# Isosurface Shift to Normal Change for SOF Instability Detection

This note turns the `alpha = 0.5` surface in SOF into a small-perturbation diagnostic that we can use to find unstable regions during training.

## 1. Surface as a level set

Let

`F(x, t) = alpha_t(x) - tau`, with `tau = 0.5`.

Then the SOF surface at training step `t` is the level set

`Sigma_t = { x | F(x, t) = 0 }`.

Write

- `g = grad_x F`
- `s = ||g||`
- `n = g / s`
- `H = Hessian_x F`
- `P = I - n n^T`

Here `n` is the unit surface normal and `P` projects onto the tangent plane.

## 2. How much the isosurface moves

Take a surface point `x` on `Sigma_t`, so `F(x, t) = 0`.

After a small training update, we get a new field `F + dF`. If we choose the minimal-motion gauge where the point only moves along the normal, then the new surface point is

`x' = x + dx`, with `dx = ds * n`.

From the level-set constraint

`F(x + dx, t + dt) = 0`

the first-order expansion gives

`g^T dx + dF = 0`.

Therefore the normal displacement is

`ds = - dF / s`

and

`dx = - (dF / s) * n`.

This is the first useful diagnostic:

`surface_shift = |dF| / (s + eps)`.

Interpretation:

- If the field value changes a lot near the `0.5` crossing, the surface moves.
- If `s` is small, even a tiny field perturbation creates a large surface shift.
- Small `s` means the `0.5` crossing is poorly conditioned, which already hints at instability.

## 3. How surface shift creates normal change

The normal is

`n = g / s`.

For a small gradient perturbation `dg_total`, the first-order normal change is

`dn = P * dg_total / s`.

If we only consider the part caused by moving to a nearby point on the same field, then

`dg_total = H dx = ds * H n`.

So the normal change caused purely by isosurface motion is

`dn_iso = ds * P H n / s`.

Substituting `ds = - dF / s` gives

`dn_iso = - dF * P H n / s^2`.

Its magnitude is

`iso_to_normal_score = |dF| * ||P H n|| / (s^2 + eps)`.

This is the exact link we want:

- `|dF| / s` tells us how much the `0.5` surface shifts.
- `||P H n|| / s` tells us how fast the normal rotates when we move along the normal direction.
- Their product tells us how much normal change is induced by the isosurface shift itself.

This means a region is unstable when either:

- the `0.5` crossing is weak, so `s` is small, or
- the local surface bends rapidly, so `||P H n||` is large, or
- both.

## 4. Full normal change between two training steps

If the field itself changes shape between two steps, not just the sampled point location, we also have a direct gradient-change term.

Let

- `DeltaF = F_{t+1}(x) - F_t(x)`
- `DeltaG = grad F_{t+1}(x) - grad F_t(x)`

Then with `ds = - DeltaF / s`, the full first-order normal change is

`dn_full ~= P (DeltaG + ds * H n) / s`.

This decomposes as:

- direct field-gradient change: `P DeltaG / s`
- shift-induced curvature term: `ds * P H n / s`

If we specifically want "isosurface movement drives normal movement", then the second term is the one to watch.

## 5. Practical scores for instability detection

At an anchor surface point, compute:

- `surface_condition = 1 / (s + eps)`
- `surface_shift = |DeltaF| / (s + eps)`
- `iso_to_normal_score = |DeltaF| * ||P H n|| / (s^2 + eps)`
- `full_normal_score = ||P (DeltaG + ds * H n)|| / (s + eps)`

Recommended interpretation:

- high `surface_shift`: the `0.5` surface is moving around
- high `iso_to_normal_score`: even small surface motion causes noticeable normal rotation
- high `full_normal_score`: the local geometry explanation is changing
- high `surface_condition`: the crossing itself is ill-conditioned

## 6. How to use this during SOF training

In SOF, the implicit field is

`F(x) = alpha(x) - 0.5`.

We can evaluate it with the same integration path used in meshing:

- [extract_mesh_tets.py](/Users/ltl/Desktop/codex_playground/SOF/extract_mesh_tets.py:29)
- [gaussian_renderer/__init__.py](/Users/ltl/Desktop/codex_playground/SOF/gaussian_renderer/__init__.py:253)

Training-time diagnostic loop:

1. Choose anchor points on the current surface.
   - back-project rendered depth samples from current train views, or
   - sample current mesh / mesh patch centers, or
   - keep a persistent anchor bank.
2. Every `K` iterations, evaluate `F`, `grad F`, and optionally `H` at those anchors.
3. Keep the previous snapshot and compute `DeltaF`, `DeltaG`.
4. Mark anchors as unstable when `surface_shift` and `iso_to_normal_score` are both high.

This is useful because:

- low-gradient `0.5` crossings correspond to thick shells and weakly localized surfaces
- high curvature amplification means small field drift leads to large normal drift
- those regions should be treated as unreliable geometry support

## 7. Cheap proxy when Hessian is too expensive

If `H` is too costly, keep a cheaper two-term proxy:

- `surface_shift = |DeltaF| / (s + eps)`
- `normal_proxy = ||P DeltaG|| / (s + eps)`

This misses the explicit curvature amplification term, but it still catches many unstable regions.

## 8. What the toy example should validate

The toy should show:

1. a local perturbation of the `0.5` surface creates a measurable `surface_shift`
2. the same perturbation causes much larger normal change where `||P H n||` is high
3. `iso_to_normal_score` lights up precisely in those unstable regions

That gives us a principled bridge from "the level set moved" to "the local geometry is unstable".
