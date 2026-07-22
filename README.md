===============================================================================
PROJECT: Artin's Constant Computation Engine
===============================================================================

OVERVIEW:
Calculates Artin's constant (C_artin ≈ 0.3739558136192022...) to arbitrary 
precision (N digits). Artin's constant represents the density of prime numbers 
for which a given integer is a primitive root.

ALGORITHM & IMPLEMENTATION:
- Riemann Zeta & Lucas Number Expansion: Converts the prime product formula 
  prod_{p} (1 - 1/(p(p-1))) into an exponentially converging series:
    ln(C_artin) = - sum_{n=2}^infty (a_n / n) * ln(zeta(n))
  where a_n is the Möbius-Lucas convolution sum_{d|n} mu(n/d) * L_d.
- Multi-core Chunking: Summation domain is partitioned across parallel worker 
  processes (gmpy2 + mpmath backend) to eliminate inter-process communication overhead.
