"""Uniform Mid-Tread / Mid-Rise Quantizer for Audio Residual Signals.

Course: ELL-786 Multimedia Systems (Assignment 3)
Author: Sudhanshu Chaudhary (2019JTM2207)
"""

import numpy as np


def compute_optimal_delta(signal: np.ndarray, bit_depth: int) -> float:
    """Computes the quantization step size Delta based on peak amplitude and bit depth."""
    max_val = float(np.max(np.abs(signal)))
    if max_val == 0:
        return 1.0
    num_levels = 2**bit_depth
    delta = (2.0 * max_val) / num_levels
    return max(delta, 1e-6)


def uniform_quantizer(
    signal: np.ndarray, bit_depth: int, delta: float | None = None
) -> tuple[np.ndarray, float]:
    """Quantizes an audio or residual difference signal into discrete integer levels.

    Args:
        signal: 1D array of input audio samples (int16 or float).
        bit_depth: Number of bits per sample (e.g. 2, 4, 8, 16).
        delta: Optional step size. If None, computed from dynamic range.

    Returns:
        q_indices: Array of quantized integer indices.
        delta: Step size used.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if delta is None:
        delta = compute_optimal_delta(signal, bit_depth)

    n_steps = 2 ** (bit_depth - 1) - 1
    abs_scaled = np.minimum(np.floor(np.abs(signal) / delta), n_steps)
    q_indices = np.sign(signal).astype(np.int32) * abs_scaled.astype(np.int32)
    return q_indices, delta


def inverse_uniform_quantizer(q_indices: np.ndarray, delta: float) -> np.ndarray:
    """Reconstructs approximate continuous values from quantized indices."""
    q_indices = np.asarray(q_indices, dtype=np.float64)
    # Mid-step reconstruction
    reconstructed = q_indices * delta + np.sign(q_indices) * (delta / 2.0)
    return reconstructed
