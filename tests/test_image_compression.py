"""Automated test suite for Multimedia Systems Image Compression (ELL-786).

Tests:
1. Repetition coding and majority-logic error correction.
2. Lossless exact arithmetic coding roundtrips.
3. 2D-DCT orthonormal basis, inverse transform fidelity, and JPEG quality scaling.
4. LZ77, LZ78, and LZW dictionary codec lossless reconstructions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO_ROOT / "image_compression"

sys.path.insert(0, str(IMAGE_DIR / "src"))

from arithmetic_codec import decode_arithmetic, encode_arithmetic
from dct_codec import (
    compress_image_dct,
    generate_dct_matrix,
    inverse_zigzag,
    zigzag_scan,
)
from dictionary_codecs import (
    lz77_decode,
    lz77_encode,
    lz78_decode,
    lz78_encode,
    lzw_decode,
    lzw_encode,
)
from repetition_codec import (
    bits_to_str,
    compute_ber,
    repetition_decode,
    repetition_encode,
    str_to_bits,
)

# ==============================================================================
# 1. Repetition Coding Tests
# ==============================================================================


def test_repetition_lossless_roundtrip():
    text = "Hello IIT Delhi!"
    bits = str_to_bits(text)
    encoded = repetition_encode(bits, r=3)
    decoded_bits = repetition_decode(encoded, r=3)
    assert decoded_bits == bits
    assert bits_to_str(decoded_bits) == text


def test_repetition_error_correction():
    text = "TEST"
    bits = str_to_bits(text)
    encoded = repetition_encode(bits, r=3)
    # Inject 1 error in every 3-bit codeword (guaranteed correctable)
    corrupted_list = list(encoded)
    for i in range(0, len(corrupted_list), 3):
        # Flip the first bit of each 3-bit block
        corrupted_list[i] = "1" if corrupted_list[i] == "0" else "0"
    corrupted = "".join(corrupted_list)

    assert compute_ber(encoded, corrupted) > 0.3
    recovered = repetition_decode(corrupted, r=3)
    assert recovered == bits
    assert bits_to_str(recovered) == text


# ==============================================================================
# 2. Arithmetic Coding Tests
# ==============================================================================


@pytest.mark.parametrize(
    "message",
    [
        "SWISS_MISS",
        "a.bar.array.by.barrayar.bay.",
        "MULTIMEDIA SYSTEMS IIT DELHI",
        "AAAAABBBBCCCCCDDE",
    ],
)
def test_arithmetic_coding_lossless(message):
    tag, length, intervals = encode_arithmetic(message)
    assert length == len(message)
    decoded = decode_arithmetic(tag, length, intervals)
    assert decoded == message


# ==============================================================================
# 3. DCT & JPEG Compression Tests
# ==============================================================================


def test_dct_matrix_orthonormality():
    """Verify T * T' == I for 8x8 DCT matrix."""
    T = generate_dct_matrix(8)
    identity = np.eye(8)
    product = T @ T.T
    np.testing.assert_allclose(product, identity, atol=1e-10)


def test_dct_zigzag_inversion():
    """Verify zigzag_scan followed by inverse_zigzag reconstructs exact block."""
    block = np.arange(64, dtype=np.float64).reshape((8, 8))
    scanned = zigzag_scan(block)
    assert len(scanned) == 64
    recovered = inverse_zigzag(scanned, n=8)
    np.testing.assert_array_equal(block, recovered)


def test_dct_image_compression_fidelity():
    """Verify DCT compression on synthetic block and sample cat.jpg."""
    # Synthetic test pattern
    synthetic = np.tile(np.linspace(20, 220, 64, dtype=np.uint8), (64, 1))
    _reconstructed, metrics = compress_image_dct(synthetic, quality=80)
    assert metrics["psnr_db"] > 35.0
    assert metrics["mse"] < 25.0

    # Test on cat.jpg if available
    cat_path = IMAGE_DIR / "cat.jpg"
    if cat_path.exists():
        import cv2

        cat_img = cv2.imread(str(cat_path), cv2.IMREAD_GRAYSCALE)
        _rec_cat, cat_metrics = compress_image_dct(cat_img, quality=75)
        assert cat_metrics["psnr_db"] > 30.0
        assert cat_metrics["zero_coefficient_percentage"] > 60.0


# ==============================================================================
# 4. Dictionary Codecs Tests (LZ77, LZ78, LZW)
# ==============================================================================


@pytest.mark.parametrize(
    "sample_text",
    [
        "a.bar.array.by.barrayar.bay.",
        "TOBEORNOTTOBEORTOBEORNOT",
        "abababababababababababab",
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    ],
)
def test_dictionary_codecs_roundtrip(sample_text):
    # LZ77
    lz77_toks = lz77_encode(sample_text)
    assert lz77_decode(lz77_toks) == sample_text

    # LZ78
    lz78_toks = lz78_encode(sample_text)
    assert lz78_decode(lz78_toks) == sample_text

    # LZW
    lzw_codes = lzw_encode(sample_text)
    assert lzw_decode(lzw_codes) == sample_text


def test_data_txt_compression():
    """Verify dictionary codecs on data.txt if present."""
    data_file = IMAGE_DIR / "data.txt"
    if data_file.exists():
        content = data_file.read_text().strip()
        if content:
            # LZW roundtrip
            codes = lzw_encode(content)
            assert lzw_decode(codes) == content
