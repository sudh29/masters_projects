"""End-to-End DPCM Audio Compression Pipeline.

Course: ELL-786 Multimedia Systems (Assignment 3)
Author: Sudhanshu Chaudhary (2019JTM2207)
"""

import math
import wave
from pathlib import Path

import numpy as np
from golomb_rice import (
    estimate_optimal_m,
    golomb_encode_stream,
    signed_to_positive_map,
)
from linear_predictor import compute_predictor_coefficients, dpcm_decode, dpcm_encode
from quantizer import inverse_uniform_quantizer, uniform_quantizer


def load_wav(filepath: str | Path) -> tuple[np.ndarray, int]:
    """Loads a 16-bit PCM WAV file and returns (audio_samples_int16, sample_rate)."""
    with wave.open(str(filepath), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    dtype = np.int16 if sampwidth == 2 else np.uint8
    data = np.frombuffer(raw_bytes, dtype=dtype)
    if n_channels > 1:
        data = data.reshape(-1, n_channels)[:, 0]  # Use first channel if stereo
    return data, framerate


def save_wav(filepath: str | Path, samples: np.ndarray, sample_rate: int):
    """Saves a 1D array of audio samples as a 16-bit mono WAV file."""
    clamped = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clamped.tobytes())


def compute_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Computes Signal-to-Noise Ratio (SNR) in decibels."""
    orig = np.asarray(original, dtype=np.float64)
    rec = np.asarray(reconstructed, dtype=np.float64)
    noise = orig - rec
    p_signal = np.sum(orig**2)
    p_noise = np.sum(noise**2)
    if p_noise <= 1e-12:
        return float("inf")
    if p_signal <= 1e-12:
        return 0.0
    return float(10.0 * math.log10(p_signal / p_noise))


def process_dpcm_pipeline(
    signal: np.ndarray,
    order: int = 2,
    bits: int = 8,
    m_param: int | None = None,
) -> dict:
    """Executes the complete DPCM + Quantization + Golomb encoding & decoding pipeline."""
    signal = np.asarray(signal, dtype=np.float64)
    n_samples = len(signal)

    # 1. Compute optimal predictor coefficients via Yule-Walker
    coefficients = compute_predictor_coefficients(signal, order=order)

    # 2. DPCM Prediction Encoding
    diff_signal, _predictions = dpcm_encode(signal, coefficients)

    # 3. Uniform Quantization of prediction residual
    q_diff, delta = uniform_quantizer(diff_signal, bit_depth=bits)

    # 4. Signed-to-positive interleaving map
    mapped_residuals = signed_to_positive_map(q_diff)

    # 5. Optimal Golomb parameter M selection
    if m_param is None:
        m_param = estimate_optimal_m(mapped_residuals)

    # 6. Golomb-Rice entropy encoding (test on first 2000 samples for speed in large audio)
    eval_count = min(n_samples, 2000)
    bitstream_sample = golomb_encode_stream(mapped_residuals[:eval_count], m_param)
    avg_bits_per_sample = len(bitstream_sample) / eval_count

    # 7. Reconstruction
    dequantized_diff = inverse_uniform_quantizer(q_diff, delta)
    reconstructed = dpcm_decode(dequantized_diff, coefficients)

    # 8. Evaluation Metrics
    snr_db = compute_snr(signal, reconstructed)
    prediction_error_energy = float(np.mean(diff_signal**2))
    original_energy = float(np.mean(signal**2))
    sper_db = 10.0 * math.log10(original_energy / max(1e-12, prediction_error_energy))

    return {
        "order": order,
        "bits": bits,
        "m_param": m_param,
        "delta": float(delta),
        "coefficients": [float(c) for c in coefficients],
        "snr_db": float(snr_db),
        "sper_db": float(sper_db),
        "avg_bits_per_sample": float(avg_bits_per_sample),
        "compression_ratio": 16.0 / max(0.1, avg_bits_per_sample),
        "reconstructed": reconstructed,
    }
