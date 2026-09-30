"""Repetition Coding and Error-Correction Module.

Course: ELL-786 Multimedia Systems (Assignment 1, Part 1)
Author: Sudhanshu Chaudhary (2019JTM2207)
"""

import random


def str_to_bits(text: str) -> str:
    """Converts a text string to an 8-bit binary string representation."""
    return "".join(f"{ord(c):08b}" for c in text)


def bits_to_str(bit_string: str) -> str:
    """Converts an 8-bit binary string back to ASCII text."""
    chars = []
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i : i + 8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return "".join(chars)


def repetition_encode(bit_string: str, r: int = 3) -> str:
    """Encodes a binary string using an (r, 1) repetition code.

    Each bit is repeated `r` times. Default r=3 produces a (3, 1) code
    capable of correcting 1 single-bit error per 3-bit codeword.
    """
    if r < 1 or r % 2 == 0:
        raise ValueError("Repetition factor r must be an odd positive integer for majority voting.")
    return "".join(b * r for b in bit_string)


def repetition_decode(encoded_string: str, r: int = 3) -> str:
    """Decodes a repetition-encoded binary string using majority logic decoding."""
    decoded = []
    threshold = r // 2
    for i in range(0, len(encoded_string), r):
        block = encoded_string[i : i + r]
        if len(block) < r:
            break
        ones = block.count("1")
        decoded.append("1" if ones > threshold else "0")
    return "".join(decoded)


def inject_hamming_errors(bit_string: str, error_count: int, seed: int | None = None) -> str:
    """Injects exactly `error_count` bit-flip errors uniformly at random."""
    if seed is not None:
        random.seed(seed)
    n = len(bit_string)
    error_count = min(error_count, n)

    bit_list = list(bit_string)
    error_indices = random.sample(range(n), error_count)
    for idx in error_indices:
        bit_list[idx] = "1" if bit_list[idx] == "0" else "0"
    return "".join(bit_list)


def compute_ber(original_bits: str, received_bits: str) -> float:
    """Computes the Bit Error Rate (BER) between two bitstrings."""
    length = min(len(original_bits), len(received_bits))
    if length == 0:
        return 0.0
    errors = sum(1 for i in range(length) if original_bits[i] != received_bits[i])
    return errors / length
