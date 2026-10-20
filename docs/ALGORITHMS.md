# Arithmetic and array mapping

The two implementations share a computation model, not Python algorithm code.
Each family has its own complete KEM implementation and independently sourced
official test fixtures.

All exported binary weight planes contain only `0` and `1`. Source matrices
and fused inverse matrices contain modular integer coefficients before bit
decomposition. Neither representation contains conductance values.

## ML-KEM

The polynomial degree is 256 and the modulus is 3329. Even and odd coefficients
form two 128-coefficient lanes. Both lanes use the same forward matrix:

```text
W[i,j] = 17^((2 * bit_reverse_7(i) + 1) * j) mod 3329
```

Here mathematical matrix rows index outputs and columns index inputs. Physical
array rows are inputs and physical columns are outputs, so the physical binary
planes represent the transpose of the mathematical matrix. This orientation
matters when exporting or inspecting CSV weights.

Each coefficient uses 12 unsigned binary weight planes. A logical 128×128
plane is split into four 64×64 tiles. Binary input bits select rows; the backend
computes binary partial counts and performs integer shift-and-add reconstruction:

```text
W x = sum over (weight_bit b, input_bit k) of
      2^(b+k) * W_bit[b] * x_bit[k].
```

For the recommended transform-only backend, forward outputs are ordinary
residues. The inverse matrix `V = W^-1 mod 3329` already includes `128^-1`.
With `R = 2^32`, inverse weights are `K = R V mod 3329`. Applying Montgomery
reduction once to the reconstructed inverse sum removes `R`; normalization must
not be applied a second time.

Quadratic base multiplication, hashing, sampling, compression, byte encoding and
the KEM rejection path remain digital. Public matrices are sampled directly in
the NTT domain; sampling them does not add a forward NTT call.

The optional signed-CBD profile uses three fixed input slots for selected
small sampled coefficients. The input sign is handled in digital accumulation;
the stored weight planes remain `0/1`.

## Saber

The native ring is `Z_(2^13)[X] / (X^256 + 1)`. LightSaber, Saber and FireSaber
use module ranks 2, 3 and 4, and sampling parameters 10, 8 and 6.
Products are evaluated under auxiliary primes 3329 and 7681.

For each prime, polynomial coefficients are interleaved into four lanes:

```text
a_lane[r][j] = a[4*j + r],  r = 0,1,2,3; j = 0,...,63
a(X) = sum over r of X^r * a_lane[r](X^4).
```

Each prime uses a shared 64×64 forward matrix: 12 bit planes for 3329 and
13 for 7681. The inverse weights include the 64-point normalization and
Montgomery scaling. Forward/inverse transform outputs use ordinary residues.
Quartic base multiplication and rank accumulation are digital operations.

Centered Chinese-remainder reconstruction recovers signed coefficients before
reduction to the native or compressed Saber modulus. For the public-by-secret
products used here, the absolute accumulated coefficient is bounded by

```text
B = rank * 256 * 4096 * mu / 2 <= 12,582,912
3329 * 7681 = 25,570,049
B < (3329 * 7681) / 2.
```

The bound depends on centered public coefficients and the secret sampling range.
The code checks the relevant bounds; two-prime reconstruction is not a general
claim for arbitrary unbounded inputs.

## What the checks measure

Official cases test the KEM byte interfaces against external expected outputs.
Transform checks test coefficient equality. Binary-array traces expose the
counts, reconstruction and reduction used to obtain those coefficients.

These are software arithmetic checks. A logical tile evaluation is not a
hardware cycle, and Python execution time is not a chip latency. This repository
does not model ADC error, read disturb, conductance distributions or physical
repair. Those experiments require separate hardware data.
