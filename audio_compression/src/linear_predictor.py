"""Linear Predictive Coding (LPC) and DPCM Differential Prediction.

Course: ELL-786 Multimedia Systems (Assignment 3)
Author: Sudhanshu Chaudhary (2019JTM2207)
"""

import numpy as np


def compute_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Computes the biased sample autocorrelation sequence for lags 0 to max_lag."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    r = np.zeros(max_lag + 1, dtype=np.float64)
    for k in range(max_lag + 1):
        r[k] = np.dot(x[: n - k], x[k:]) / n
    return r


def compute_predictor_coefficients(signal: np.ndarray, order: int = 2) -> np.ndarray:
    """Solves the Yule-Walker equations R * A = r to find optimal prediction coefficients.

    Args:
        signal: 1D audio sample array.
        order: Linear prediction filter order (e.g. 1, 2, 3, 4).

    Returns:
        A: 1D array of prediction coefficients [a_1, a_2, ..., a_p].
    """
    if order < 1:
        raise ValueError("Predictor order must be at least 1.")

    autocorr = compute_autocorrelation(signal, order)
    r0 = autocorr[0]
    if r0 == 0:
        return np.zeros(order, dtype=np.float64)

    # Toeplitz autocorrelation matrix R
    R = np.zeros((order, order), dtype=np.float64)
    for i in range(order):
        for j in range(order):
            R[i, j] = autocorr[abs(i - j)]

    # Cross-correlation vector r
    r = autocorr[1 : order + 1]

    # Add minor Tikhonov regularization for numerical stability if singular
    R += np.eye(order) * 1e-9

    try:
        A = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(R) @ r

    return A


def dpcm_encode(signal: np.ndarray, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes predicted samples and prediction residual difference.

    Args:
        signal: 1D input audio signal.
        coefficients: Predictor coefficients [a_1, a_2, ..., a_p].

    Returns:
        diff_signal: Residual error e(n) = x(n) - x_hat(n).
        predictions: Predicted signal x_hat(n).
    """
    signal = np.asarray(signal, dtype=np.float64)
    order = len(coefficients)
    n_samples = len(signal)

    predictions = np.zeros(n_samples, dtype=np.float64)
    diff_signal = np.zeros(n_samples, dtype=np.float64)

    diff_signal[0] = signal[0]
    predictions[0] = 0.0

    for n in range(1, n_samples):
        # x_hat(n) = sum_{k=1}^{min(n, order)} a_k * x(n - k)
        pred = 0.0
        limit = min(n, order)
        for k in range(limit):
            pred += coefficients[k] * signal[n - 1 - k]
        predictions[n] = pred
        diff_signal[n] = signal[n] - pred

    return diff_signal, predictions


def dpcm_decode(diff_signal: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Reconstructs the original audio signal from the residual difference.

    Args:
        diff_signal: Residual error sequence e(n).
        coefficients: Predictor coefficients [a_1, a_2, ..., a_p].

    Returns:
        reconstructed: Reconstructed audio signal.
    """
    diff_signal = np.asarray(diff_signal, dtype=np.float64)
    order = len(coefficients)
    n_samples = len(diff_signal)

    reconstructed = np.zeros(n_samples, dtype=np.float64)
    reconstructed[0] = diff_signal[0]

    for n in range(1, n_samples):
        pred = 0.0
        limit = min(n, order)
        for k in range(limit):
            pred += coefficients[k] * reconstructed[n - 1 - k]
        reconstructed[n] = diff_signal[n] + pred

    return reconstructed
