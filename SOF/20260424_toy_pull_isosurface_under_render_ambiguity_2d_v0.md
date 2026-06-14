# 2D Toy: Pulling a Wrong Front Isosurface While Render Stays Correct

This is the 2D version of the previous 1D ambiguity toy.

We work in an `x-z` slice:

- `x`: tangential direction along the surface
- `z`: depth / local normal direction

## Field

Define a local opacity-density field

`sigma(x, z) = A(x) * exp(-(z - c(x))^2 / (2 w^2))`

where:

- `A(x)` is a per-column amplitude profile
- `c(x)` is the center along depth
- `w` is a constant thickness

## Render proxy

At each `x`, define a 1D render proxy by integrating along depth:

`R(x) = 1 - exp(- integral sigma(x, z) dz )`

For the Gaussian above:

`integral sigma(x, z) dz = A(x) * w * sqrt(2 pi)`

so

`R(x) = 1 - exp(-A(x) w sqrt(2 pi))`

This means:

- `R(x)` depends on `A(x)` and `w`
- `R(x)` does **not** depend on `c(x)`

So we can move the whole local density blob forward or backward along `z` without changing the rendered signal.

## Front isosurface

Take the front crossing of a fixed level `sigma = sigma_iso`.

For each `x`, the front crossing is

`z_front(x) = c(x) - w * sqrt(2 log(A(x) / sigma_iso))`

This is the extracted surface in the toy.

## Ambiguity construction

Choose a true front surface `z_true(x)`.

Set the correct center field to

`c_true(x) = z_true(x) + w * sqrt(2 log(A(x) / sigma_iso))`

Then create a wrong but render-equivalent field:

`c_wrong(x) = c_true(x) + a * bump(x)`

where `bump(x)` is a localized positive bump and `a > 0`.

Consequences:

- the rendered profile `R(x)` is unchanged for all `a`
- the extracted front surface becomes

`z_front_wrong(x) = z_true(x) + a * bump(x)`

So the front surface is locally wrong even though the render is still perfect.

## Manual pull

The toy manually reduces the bump coefficient `a` toward `0`.

This is exactly the operation we want to reason about:

- can we drag a wrong front isosurface back to the true surface
- while leaving the rendered signal unchanged

The answer in this toy is yes.

But the toy also shows the limitation:

- render alone does not tell us which `c(x)` is correct
- the pull signal is what chooses the right member of the render-equivalent family
