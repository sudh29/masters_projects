"""Exact Arithmetic Coding (Lossless Entropy Coding).

Course: ELL-786 Multimedia Systems (Assignment 1, Part 1)
Author: Sudhanshu Chaudhary (2019JTM2207)
"""

from collections import Counter
from fractions import Fraction


def build_cumulative_intervals(data: str) -> dict[str, tuple[Fraction, Fraction]]:
    """Builds normalized cumulative probability intervals for symbols in `data`."""
    counts = Counter(data)
    total = len(data)
    sorted_symbols = sorted(counts.keys())

    intervals = {}
    current_low = Fraction(0, 1)
    for sym in sorted_symbols:
        prob = Fraction(counts[sym], total)
        current_high = current_low + prob
        intervals[sym] = (current_low, current_high)
        current_low = current_high
    return intervals


def encode_arithmetic(data: str) -> tuple[Fraction, int, dict[str, tuple[Fraction, Fraction]]]:
    """Encodes a string using interval subdivision.

    Returns:
        tag: A Fraction in [0, 1) representing the message.
        length: Number of symbols in the encoded string.
        intervals: Symbol interval mapping required for decoding.
    """
    if not data:
        return Fraction(0, 1), 0, {}

    intervals = build_cumulative_intervals(data)
    low = Fraction(0, 1)
    high = Fraction(1, 1)

    for sym in data:
        sym_low, sym_high = intervals[sym]
        range_width = high - low
        high = low + range_width * sym_high
        low = low + range_width * sym_low

    tag = (low + high) / 2
    return tag, len(data), intervals


def decode_arithmetic(
    tag: Fraction, length: int, intervals: dict[str, tuple[Fraction, Fraction]]
) -> str:
    """Decodes an arithmetic tag back into the original string."""
    if length == 0 or not intervals:
        return ""

    low = Fraction(0, 1)
    high = Fraction(1, 1)
    decoded_chars = []

    for _ in range(length):
        range_width = high - low
        # Find which symbol's sub-interval contains tag
        target = (tag - low) / range_width
        matched_sym = None
        for sym, (s_low, s_high) in intervals.items():
            if s_low <= target < s_high:
                matched_sym = sym
                break
        if matched_sym is None:
            # Fallback to the last symbol due to fractional boundary
            matched_sym = max(intervals.keys(), key=lambda s: intervals[s][1])

        decoded_chars.append(matched_sym)
        sym_low, sym_high = intervals[matched_sym]
        high = low + range_width * sym_high
        low = low + range_width * sym_low

    return "".join(decoded_chars)
