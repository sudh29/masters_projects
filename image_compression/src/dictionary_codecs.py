"""Dictionary-Based Lossless Data Compression Algorithms (LZ77, LZ78, LZW).

Course: ELL-786 Multimedia Systems (Assignment 2)
Author: Sudhanshu Chaudhary (2019JTM2207)
"""

# ==============================================================================
# 1. LZ77 Algorithm (Sliding Window)
# ==============================================================================


def lz77_encode(
    data: str, search_window: int = 32, lookahead_buffer: int = 16
) -> list[tuple[int, int, str]]:
    """Encodes a string using the LZ77 sliding window algorithm.

    Tokens produced: (offset, length, next_char)
    """
    tokens = []
    cursor = 0
    n = len(data)

    while cursor < n:
        best_offset = 0
        best_length = 0

        start_search = max(0, cursor - search_window)
        search_buf = data[start_search:cursor]

        # Find longest match in search buffer that matches beginning of lookahead
        max_match_len = min(lookahead_buffer, n - cursor - 1)
        for length in range(1, max_match_len + 1):
            sub = data[cursor : cursor + length]
            pos = search_buf.rfind(sub)
            if pos != -1:
                best_offset = len(search_buf) - pos
                best_length = length
            else:
                break

        cursor += best_length
        next_char = data[cursor] if cursor < n else ""
        cursor += 1
        tokens.append((best_offset, best_length, next_char))

    return tokens


def lz77_decode(tokens: list[tuple[int, int, str]]) -> str:
    """Decodes a list of LZ77 tokens back into the original string."""
    output = []
    for offset, length, next_char in tokens:
        if length > 0:
            start = len(output) - offset
            for i in range(length):
                output.append(output[start + i])
        if next_char:
            output.append(next_char)
    return "".join(output)


# ==============================================================================
# 2. LZ78 Algorithm (Dynamic Dictionary)
# ==============================================================================


def lz78_encode(data: str) -> list[tuple[int, str]]:
    """Encodes a string using the LZ78 algorithm.

    Tokens produced: (dictionary_index, next_char)
    """
    dictionary = {"": 0}
    tokens = []
    w = ""
    for char in data:
        wc = w + char
        if wc in dictionary:
            w = wc
        else:
            tokens.append((dictionary[w], char))
            dictionary[wc] = len(dictionary)
            w = ""
    if w:
        # Final token if input ends with a matched phrase
        prefix = w[:-1]
        last_char = w[-1]
        tokens.append((dictionary[prefix], last_char))
    return tokens


def lz78_decode(tokens: list[tuple[int, str]]) -> str:
    """Decodes a list of LZ78 tokens back into the original string."""
    dictionary = {0: ""}
    output = []
    for idx, char in tokens:
        phrase = dictionary[idx] + char
        output.append(phrase)
        dictionary[len(dictionary)] = phrase
    return "".join(output)


# ==============================================================================
# 3. LZW Algorithm (Lempel-Ziv-Welch)
# ==============================================================================


def lzw_encode(data: str, initial_dict: dict[str, int] | None = None) -> list[int]:
    """Encodes a string using the classic LZW algorithm."""
    if not data:
        return []

    if initial_dict is None:
        dictionary = {chr(i): i for i in range(256)}
    else:
        dictionary = dict(initial_dict)

    w = ""
    result = []
    for c in data:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = len(dictionary)
            w = c
    if w:
        result.append(dictionary[w])
    return result


def lzw_decode(codes: list[int], initial_dict: dict[int, str] | None = None) -> str:
    """Decodes a list of LZW codes back into the original string."""
    if not codes:
        return ""

    if initial_dict is None:
        dictionary = {i: chr(i) for i in range(256)}
    else:
        dictionary = dict(initial_dict)

    w = dictionary[codes[0]]
    result = [w]

    for k in codes[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == len(dictionary):
            # Special edge case: cScSc pattern
            entry = w + w[0]
        else:
            raise ValueError(f"Bad compressed code in LZW stream: {k}")

        result.append(entry)
        dictionary[len(dictionary)] = w + entry[0]
        w = entry

    return "".join(result)
