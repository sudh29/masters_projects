"""Pattern matching algorithm for Raspberry Pi Lab (Assignment 8).

Implements sequence matching of sub-patterns across input bit sequences.
"""


def pattern_match(pattern: list, sequence: list) -> int:
    """Counts occurrences of `pattern` in `sequence` advancing by `len(pattern)`.

    Args:
        pattern: The sub-sequence to search for.
        sequence: The total sequence being examined.

    Returns:
        Number of matches found.
    """
    len_p = len(pattern)
    len_s = len(sequence)
    if len_p == 0 or len_s < len_p:
        return 0

    match_count = 0
    j = 0
    while j + len_p <= len_s:
        chunk = sequence[j : j + len_p]
        if chunk == pattern:
            match_count += 1
            j += len_p
        else:
            j += 1
    return match_count


def string_to_bit_list(text: str) -> list[int]:
    """Encodes a string into a list of binary integers (8 bits per character)."""
    bits = []
    for byte in text.encode("utf-8"):
        for b in f"{byte:08b}":
            bits.append(int(b))
    return bits


def bit_list_to_string(bits: list[int]) -> str:
    """Decodes a list of 8-bit integers back into a string."""
    chars = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i : i + 8]
        if len(byte_bits) == 8:
            byte_val = int("".join(str(b) for b in byte_bits), 2)
            chars.append(chr(byte_val))
    return "".join(chars)
