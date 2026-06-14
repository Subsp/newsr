# 2D Toy: Membrane Shrink on a Smooth Patch

This toy targets the failure mode:

- the learned surface is already thin
- boundaries and hole rims are well constrained
- smooth interior regions still cave inward

That is not a "thick shell" problem.
It is a "thin but biased membrane" problem.

## Setup

Work with a height field `z(x, y)` over a 2D patch.

- true surface: `z_true(x, y) = 0`
- strong support only on:
  - outer boundary
  - hole boundaries
- smooth interior has weak or no direct positional support

We optimize a quadratic energy:

`E(z) = 0.5 * lambda_anchor * sum_A z^2
      + 0.5 * lambda_pull   * sum_P z^2
      + 0.5 * lambda_smooth * sum_(i,j) (z_i - z_j)^2
      + lambda_pressure * sum_Omega z`

where:

- `A` is the boundary / hole-rim anchor set
- `P` is an optional interior pull set
- `Omega` is the whole patch

Interpretation:

- `lambda_anchor`: strong local data support near feature edges
- `lambda_smooth`: thin smooth membrane prior
- `lambda_pressure`: compactness / shrink / inward bias
- `lambda_pull`: explicit position-restoring signal

## Why the interior caves in

Ignoring masks for a moment, the Euler-Lagrange equation is:

`-lambda_smooth * Delta z + lambda_anchor * w_A z + lambda_pull * w_P z + lambda_pressure = 0`

In the smooth interior, `w_A` is near zero, so without pull we approximately get:

`-lambda_smooth * Delta z + lambda_pressure = 0`

or

`Delta z = lambda_pressure / lambda_smooth`

With zero boundary anchors, this produces a concave negative bowl.

So the toy says:

- sparse edge support alone does not make the interior wrong
- the interior caves in when there is also a shrink / compactness bias

## What the toy should show

1. With anchors only and no pressure, the flat patch is recovered.
2. Adding pressure creates a smooth inward sag between anchor-rich regions.
3. Adding an interior position pull suppresses the sag without changing the anchor behavior.

This is the closest toy abstraction of:

- "the patch is thin"
- "feature edges are aligned"
- "smooth areas sink inward"
