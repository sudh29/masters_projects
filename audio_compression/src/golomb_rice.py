"""Golomb-Rice Entropy Coding for Geometric Residual Distributions.

Course: ELL-786 Multimedia Systems (Assignment 3)
Author: Sudhanshu Chaudhary (2019JTM2207)
"""

import math

import numpy as np


def signed_to_positive_map(values: np.ndarray) -> np.ndarray:
    """Interleaving map from signed integers to non-negative integers.

    x < 0 -> 2 * |x| - 1  (mapped to odd integers: -1 -> 1, -2 -> 3, -3 -> 5)
    x >= 0 -> 2 * x       (mapped to even integers: 0 -> 0, 1 -> 2, 2 -> 4)
    """
    values = np.asarray(values, dtype=np.int64)
    mapped = np.where(values < 0, 2 * np.abs(values) - 1, 2 * values)
    return mapped


def positive_to_signed_map(mapped_values: np.ndarray) -> np.ndarray:
    """Inverse mapping from non-negative integers back to signed values."""
    mapped_values = np.asarray(mapped_values, dtype=np.int64)
    signed = np.where(
        mapped_values % 2 == 1,
        -((mapped_values + 1) // 2),
        mapped_values // 2,
    )
    return signed


def estimate_optimal_m(mapped_values: np.ndarray) -> int:
    """Estimates the optimal Golomb parameter M based on the mean of the geometric distribution."""
    if len(mapped_values) == 0:
        return 1
    mean_val = float(np.mean(mapped_values))
    if mean_val <= 0:
        return 1
    # For geometric distribution with mean mu: p = 1 / (1 + mu)
    # Optimal m = ceil(-ln(2) / ln(1 - 1/(1+mu)))
    p = 1.0 / (1.0 + mean_val)
    if p >= 1.0:
        return 1
    m = math.ceil(-math.log(2.0) / math.log(1.0 - p))
    return max(1, m)


def golomb_encode_value(n: int, m: int) -> str:
    """Encodes a single non-negative integer n using Golomb coding with parameter m."""
    if n < 0:
        raise ValueError("Golomb coding only accepts non-negative integers.")
    if m < 1:
        raise ValueError("Golomb parameter m must be >= 1.")

    q = n // m
    r = n % m

    # Unary code for quotient: q '1's followed by a '0'
    unary = "1" * q + "0"

    # Truncated binary code for remainder r
    b = math.ceil(math.log2(m)) if m > 1 else 0
    if b == 0:
        remainder_code = ""
    else:
        cutoff = (2**b) - m
        if r < cutoff:
            remainder_code = format(r, f"0{b - 1}b") if (b - 1) > 0 else ""
        else:
            remainder_code = format(r + cutoff, f"0{b}b")

    return unary + remainder_code


def golomb_decode_value(bitstream: str, start_idx: int, m: int) -> tuple[int, int]:
    """Decodes a single Golomb-coded value from bitstream starting at start_idx.

    Returns:
        (value, next_idx)
    """
    if m < 1:
        raise ValueError("Golomb parameter m must be >= 1.")

    # Read unary quotient: count 1s until 0
    q = 0
    curr = start_idx
    while curr < len(bitstream) and bitstream[curr] == "1":
        q += 1
        curr += 1
    if curr >= len(bitstream):
        raise ValueError("Truncated bitstream during Golomb quotient read.")
    curr += 1  # Skip terminating '0'

    # Read truncated binary remainder
    b = math.ceil(math.log2(m)) if m > 1 else 0
    if b == 0:
        r = 0
    else:
        cutoff = (2**b) - m
        if b - 1 > 0:
            first_bits = int(bitstream[curr : curr + (b - 1)], 2)
            curr += b - 1
            if first_bits < cutoff:
                r = first_bits
            else:
                next_bit = int(bitstream[curr], 2)
                curr += 1
                r = (first_bits << 1) + next_bit - cutoff
        else:
            r = int(bitstream[curr], 2)
            curr += 1

    n = q * m + r
    return n, curr


def golomb_encode_stream(mapped_values: np.ndarray, m: int) -> str:
    """Encodes an array of non-negative integers into a continuous bitstring."""
    return "".join(golomb_encode_value(int(v), m) for v in mapped_values)


def golomb_decode_stream(bitstream: str, count: int, m: int) -> list[int]:
    """Decodes a continuous bitstring into an array of count integers."""
    result = []
    idx = 0
    for _ in range(count):
        val, idx = golomb_decode_value(bitstream, idx, m)
        result.append(val)
    return result
