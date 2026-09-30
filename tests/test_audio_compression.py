"""Automated test suite for Multimedia Systems Audio Compression (ELL-786 Assignment 3).

Tests:
1. Uniform quantizer and dequantizer accuracy.
2. Yule-Walker LPC predictor coefficient stability.
3. DPCM lossless encode/decode roundtrip without quantization.
4. Golomb-Rice non-negative mapping and lossless stream encoding/decoding.
5. End-to-end WAV compression fidelity and SNR monotonic scaling.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "audio_compression"

sys.path.insert(0, str(AUDIO_DIR / "src"))

from audio_pipeline import load_wav, process_dpcm_pipeline
from golomb_rice import (
    estimate_optimal_m,
    golomb_decode_stream,
    golomb_decode_value,
    golomb_encode_stream,
    golomb_encode_value,
    positive_to_signed_map,
    signed_to_positive_map,
)
from linear_predictor import compute_predictor_coefficients, dpcm_decode, dpcm_encode
from quantizer import compute_optimal_delta, inverse_uniform_quantizer, uniform_quantizer

# ==============================================================================
# 1. Quantizer Tests
# ==============================================================================


def test_quantizer_step_size():
    signal = np.array([-100.0, 50.0, 100.0])
    delta = compute_optimal_delta(signal, bit_depth=4)  # 16 levels -> 200 / 16 = 12.5
    assert abs(delta - 12.5) < 1e-4


def test_quantizer_reconstruction_error():
    signal = np.linspace(-500, 500, 101)
    q_indices, delta = uniform_quantizer(signal, bit_depth=8)
    reconstructed = inverse_uniform_quantizer(q_indices, delta)
    errors = np.abs(signal - reconstructed)
    # Maximum error in uniform quantizer is bounded by Delta
    assert np.max(errors) <= delta


# ==============================================================================
# 2. Linear Prediction Tests
# ==============================================================================


def test_lpc_coefficients_calculation():
    # Synthetic decaying sinusoids (autoregressive process)
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    coeffs = compute_predictor_coefficients(signal, order=2)
    assert len(coeffs) == 2
    assert not np.isnan(coeffs).any()


def test_dpcm_lossless_roundtrip_without_quantization():
    signal = np.array([10.0, 25.0, 30.0, 45.0, 40.0, 35.0, 20.0])
    coeffs = np.array([0.8, -0.2])
    diff, _ = dpcm_encode(signal, coeffs)
    reconstructed = dpcm_decode(diff, coeffs)
    np.testing.assert_allclose(signal, reconstructed, atol=1e-10)


# ==============================================================================
# 3. Golomb-Rice Coding Tests
# ==============================================================================


def test_signed_to_positive_mapping():
    inputs = np.array([0, -1, 1, -2, 2, -3, 3])
    expected = np.array([0, 1, 2, 3, 4, 5, 6])
    mapped = signed_to_positive_map(inputs)
    np.testing.assert_array_equal(mapped, expected)
    recovered = positive_to_signed_map(mapped)
    np.testing.assert_array_equal(recovered, inputs)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 16])
def test_golomb_value_roundtrip(m):
    for val in range(25):
        bitstream = golomb_encode_value(val, m)
        recovered, next_idx = golomb_decode_value(bitstream, 0, m)
        assert recovered == val
        assert next_idx == len(bitstream)


def test_golomb_stream_roundtrip():
    values = np.array([0, 1, 4, 12, 3, 0, 2, 7, 15, 1, 0, 5])
    m = estimate_optimal_m(values)
    bitstream = golomb_encode_stream(values, m)
    recovered = golomb_decode_stream(bitstream, len(values), m)
    np.testing.assert_array_equal(values, recovered)


# ==============================================================================
# 4. End-to-End Pipeline on Audio Files
# ==============================================================================


def test_audio_pipeline_execution():
    wav_path = AUDIO_DIR / "1Dialogue.wav"
    assert wav_path.exists()
    samples, sr = load_wav(wav_path)
    assert len(samples) > 0
    assert sr > 0

    # Test order 2 with 8 bits
    res = process_dpcm_pipeline(samples[:4000], order=2, bits=8)
    assert res["snr_db"] > 15.0
    assert len(res["reconstructed"]) == 4000
    assert res["avg_bits_per_sample"] > 0


def test_snr_increases_with_bit_depth():
    wav_path = AUDIO_DIR / "1Dialogue.wav"
    samples, _ = load_wav(wav_path)
    sub = samples[:3000]

    res_4bit = process_dpcm_pipeline(sub, order=2, bits=4)
    res_8bit = process_dpcm_pipeline(sub, order=2, bits=8)
    res_12bit = process_dpcm_pipeline(sub, order=2, bits=12)

    # More bits -> finer quantization -> higher SNR
    assert res_8bit["snr_db"] > res_4bit["snr_db"]
    assert res_12bit["snr_db"] > res_8bit["snr_db"]
