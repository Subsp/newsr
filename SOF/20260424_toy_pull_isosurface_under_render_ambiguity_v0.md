# Toy: Pulling a Wrong Isosurface Under Render Ambiguity

This toy isolates the case:

- rendered appearance is already correct
- the extracted isosurface is still offset from the true surface
- we manually add a "pull the isosurface here" signal

The goal is to answer a narrow question:

Can a wrong but render-perfect field be moved back to the right surface without breaking render consistency?

## Toy field

We use a 1D depth-axis field

`sigma(z) = A * exp(-(z - c)^2 / (2 w^2))`

where:

- `A` is amplitude
- `c` is the center along depth
- `w` is thickness

This behaves like a local thick shell or opacity blob along one normal/ray direction.

## Render proxy

Use a constant-color transmittance proxy:

`R = 1 - exp(- integral sigma(z) dz )`

For the Gaussian above:

`integral sigma(z) dz = A * w * sqrt(2 pi)`

so

`R = 1 - exp(-A w sqrt(2 pi))`

Important consequence:

`R` depends on `A w`, but not on `c`.

So if we slide the whole blob forward or backward in depth, the render proxy is unchanged.

That is the toy ambiguity: a field can render perfectly while its extracted surface moves around.

## Extracted surface

Use the front crossing of a fixed isovalue `sigma(z) = sigma_iso`.

If `A > sigma_iso`, the front crossing is

`z_iso = c - w * sqrt(2 log(A / sigma_iso))`

This is the "mesh" location in the toy.

## What the toy tests

1. Start from a wrong solution:
   - `R` already matches the target exactly
   - `z_iso` is offset from the true surface
2. Add a manual pull term:
   - `L_pull = (z_iso - z_true)^2`
3. Optimize:
   - `L = lambda_render * (R - R_target)^2 + lambda_pull * (z_iso - z_true)^2 + weak_shape_reg`

## What we expect

- with render loss alone, nothing moves because render is already perfect
- with render plus pull, the optimizer can recover a correct-surface solution
- but the solution is not unique, because many fields can preserve the same render while placing the isosurface differently

So this toy is useful if we want to reason about:

- why photometric correctness alone is insufficient
- when a direct surface-pull signal can work
- why a pull signal usually needs a second bias or regularizer to choose among many render-equivalent fields
