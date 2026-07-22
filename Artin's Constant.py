#!/usr/bin/env python3
r"""
Artin's Constant Calculator (High-Performance Computing Edition)
--------------------------------------------------------------
This script calculates Artin's Constant to a user-specified [N] significant digits.

Mathematical Approach:
Artin's constant is defined as the infinite product over all primes:
    C_artin = \prod_{p \in \mathbb{P}} \left(1 - \frac{1}{p(p-1)}\right)

Directly evaluating the prime product is computationally infeasible for high precision.
Instead, we expand the product into an infinite series using the Riemann Zeta function 
and Lucas numbers, derived via Taylor expansion and Möbius inversion of the Prime Zeta function:
    \ln(C_artin) = - \sum_{n=2}^{\infty} \frac{a_n}{n} \ln(\zeta(n))
where:
    a_n = \sum_{d | n} \mu(n/d) L_d
    \mu(x) is the Möbius function
    L_d is the d-th Lucas number

This series converges exponentially fast, requiring only M ≈ N * log_2(10) terms 
to reach N digits of precision.

Chunking Strategy & Parallelization:
To compute this in parallel on exactly 12 cores, the summation domain [2, M] is 
divided into 12 contiguous mathematical intervals (chunks). Each chunk is 
processed independently by a worker process in a `multiprocessing.Pool(12)`.
The partial sums are computed locally, avoiding inter-process communication overhead.
Once all workers finish, the main process aggregates the 12 partial sums and exponentiates 
the final value to yield Artin's Constant.

Memory Management:
Generators are used within worker processes to yield individual series terms dynamically, 
preventing the storage of large structures in RAM. `gc.collect()` is explicitly called 
after significant aggregation steps to ensure the heap remains minimal.
"""

import sys
import math
import argparse
import multiprocessing
import gc
import functools
import os

# Enforce gmpy2 backend for mpmath to ensure maximum C-level optimization
os.environ['MPMATH_GMPY2'] = '1'
import gmpy2
import mpmath

# Raise the limit for integer-to-string conversions for extreme precision targets
sys.set_int_max_str_digits(0)

def get_a_array(limit):
    """
    Precomputes the convolution coefficient a_n = Sum_{d | n} [ mu(n/d) * L_d ]
    for all n <= limit using a highly optimized sieve approach.
    This avoids slow trial division and lru_cache overhead across workers.
    """
    # 1. Precompute Lucas numbers up to limit
    lucas = [2, 1] + [0] * (limit - 1)
    for i in range(2, limit + 1):
        lucas[i] = lucas[i - 1] + lucas[i - 2]
        
    # 2. Sieve for Möbius function
    moebius = [0] * (limit + 1)
    moebius[1] = 1
    for i in range(1, limit + 1):
        if moebius[i]:
            for j in range(i * 2, limit + 1, i):
                moebius[j] -= moebius[i]
                
    # 3. Sieve for a_n
    a = [0] * (limit + 1)
    for d in range(1, limit + 1):
        L_d = lucas[d]
        if L_d == 0: continue
        for n in range(d, limit + 1, d):
            m = moebius[n // d]
            if m:
                a[n] += m * L_d
                
    return a

def worker_partial_sum(start_n, end_n, dps_precision):
    """
    Worker function to compute the partial sum of the series for the interval [start_n, end_n).
    
    Args:
        start_n (int): Starting index (inclusive) of the chunk.
        end_n (int): Ending index (exclusive) of the chunk.
        dps_precision (int): Exact decimal precision to configure within the worker.
        
    Returns:
        str: String representation of the partial sum to safely transmit the 
             arbitrary precision float back to the parent process without IPC pickling precision drift.
    """
    # Configure worker-local precision
    mpmath.mp.dps = dps_precision
    
    # Precompute all a_n values needed for this chunk
    a = get_a_array(end_n - 1)
    
    partial_sum = mpmath.mpf(0)
    
    # Generator to evaluate the mathematical chunk without holding history in memory
    def term_generator():
        for n in range(start_n, end_n):
            a_n = a[n]
            if a_n != 0:
                # Dynamic dps adjustment for ln(zeta(n)) to prevent catastrophic cancellation
                # mpmath.zeta(n) ~ 1 + 2^-n, which loses ~ n * log10(2) digits of relative precision.
                extra = int(n * 0.30103) + 10
                with mpmath.workdps(dps_precision + extra):
                    z = mpmath.zeta(n)
                    res = mpmath.log(z)
                # The result 'res' is safely rounded down to dps_precision relative digits,
                # which keeps more than enough absolute precision for the sum.
                yield (mpmath.mpf(a_n) / mpmath.mpf(n)) * res
                
    for term in term_generator():
        partial_sum += term
        
    # Explicit garbage collection to release fragmented memory from large mpf allocations
    del a
    gc.collect()
    
    # Return as string with extended precision to guarantee no bits are lost during transfer
    return mpmath.nstr(partial_sum, dps_precision + 10, min_fixed=-mpmath.inf, max_fixed=mpmath.inf)

def calculate_artins_constant(N, num_cores=12):
    """
    Main orchestration function. Configures precision, partitions the mathematical domain,
    spawns workers, aggregates results, and performs strict digit truncation.
    """
    # 1. Memory and Precision Configuration
    # Set internal working precision at least 50 digits higher than target N 
    # to completely eliminate floating-point drift and hardware rounding errors during aggregation.
    working_dps = N + 50
    mpmath.mp.dps = working_dps
    
    # 2. Mathematical Domain Calculation
    # Since ln(zeta(n)) decays as ~2^-n, but a_n grows as ~phi^n, 
    # the sequence decays as ~(phi/2)^n where phi = (1 + sqrt(5))/2.
    # To reach an absolute error of 10^(-working_dps), we need n to reach working_dps * log10(2/phi).
    phi = (1 + math.sqrt(5)) / 2
    M = int(working_dps * math.log(10) / math.log(2 / phi)) + 10
    
    print(f"[*] Target Precision: {N} significant digits")
    print(f"[*] Internal Working Precision: {working_dps} digits (Target + 50 guard digits)")
    print(f"[*] Series Convergence Threshold (M): {M} terms")
    print(f"[*] Multiprocessing Workers: {num_cores} CPU cores")
    
    # 3. Interval Chunking Logic
    # Distribute [2, M] evenly across the exactly specified 12 worker processes.
    domain_size = M - 1 # Since we start at n=2
    chunk_size = math.ceil(domain_size / num_cores)
    
    chunks = []
    current = 2
    for _ in range(num_cores):
        end = min(current + chunk_size, M + 1)
        if current < end:
            chunks.append((current, end, working_dps))
        current = end
        
    print(f"[*] Dispatched {len(chunks)} computational chunks to pool.")

    # 4. Explicit Parallelization
    # Utilizing exactly 12 CPU cores as architecturally mandated.
    with multiprocessing.Pool(processes=num_cores) as pool:
        results_str = pool.starmap(worker_partial_sum, chunks)
        
    print("[*] All mathematical chunks computed. Aggregating partial sums...")
    
    # 5. Final Aggregation
    total_sum = mpmath.mpf(0)
    for res_str in results_str:
        total_sum += mpmath.mpf(res_str)
        
    # Exponentiate the negative sum to resolve Artin's Constant
    # C_artin = exp( - sum )
    artin_constant = mpmath.exp(-total_sum)
    
    # Free memory of large strings and intermediate sums
    del results_str
    gc.collect()
    
    print("[*] Performing strict truncation for OEIS submission formatting...")
    
    # 6. Truncation and Formatting
    # Convert to string with guard digits to avoid implicitly rounding up the target Nth digit
    out_str = mpmath.nstr(artin_constant, N + 10, min_fixed=-mpmath.inf, max_fixed=mpmath.inf, strip_zeros=False)
    
    # Extract only the fractional decimal digits (Artin's constant is ~0.373955...)
    # The string will be formatted identically to "0.3739558136..."
    if out_str.startswith("0."):
        digits_only = out_str[2:]
    else:
        # Theoretical mathematical fallback 
        digits_only = out_str.replace(".", "")
        
    # Strictly truncate (round down) at the N-th digit, completely removing trailing guard numbers
    final_digits = digits_only[:N]
    
    return final_digits

def output_oeis_b_file(digits, filename):
    """
    Secondary function: Outputs a standard OEIS b-file format.
    Format: [index] [digit] separated by a space on each line.
    """
    with open(filename, 'w') as f:
        # Artin's Constant decimal expansion (OEIS A005596) sets the first significant digit to index 1.
        for index, digit in enumerate(digits, start=1):
            f.write(f"{index} {digit}\n")

if __name__ == '__main__':
    # Parse N from command line arguments
    parser = argparse.ArgumentParser(description="Calculate Artin's Constant to exact [N] significant digits via Multiprocessing.")
    parser.add_argument('N', type=int, nargs='?', default=3000, help='Number of exact significant digits to calculate (default: 1000)')
    args = parser.parse_args()
    
    target_N = args.N
    
    # Execute the HPC mathematical calculation
    final_digits = calculate_artins_constant(target_N, num_cores=12)
    
    # Define dynamic output filenames based on N
    txt_filename = f"Artins_Constant_{target_N}_digits.txt"
    b_file_filename = f"b_file_Artins_Constant_{target_N}.txt"
    
    # 1. Output the single continuous string (Primary constraint)
    with open(txt_filename, 'w') as f:
        f.write(final_digits)
    print(f"[+] SUCCESS: Saved continuous digit string to {txt_filename}")
    
    # 2. Output the standard OEIS b-file (Secondary constraint)
    output_oeis_b_file(final_digits, b_file_filename)
    print(f"[+] SUCCESS: Saved OEIS b-file format to {b_file_filename}")
