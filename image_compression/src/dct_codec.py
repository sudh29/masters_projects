"""2D Discrete Cosine Transform (DCT) and JPEG-style Image Compression Codec.

Course: ELL-786 Multimedia Systems (Assignment 1, Part 2)
Author: Sudhanshu Chaudhary (2019JTM2207)
"""

import math

import numpy as np

# Standard JPEG Luminance Quantization Table (8x8)
JPEG_LUMA_QUANT = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)


def generate_dct_matrix(n: int = 8) -> np.ndarray:
    """Generates an N x N orthonormal DCT-II transformation matrix."""
    T = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            angle = ((2 * j + 1) * i * math.pi) / (2 * n)
            if i == 0:
                T[i, j] = 1.0 / math.sqrt(n)
            else:
                T[i, j] = math.sqrt(2.0 / n) * math.cos(angle)
    return T


def get_scaled_quant_matrix(quality: int = 50) -> np.ndarray:
    """Scales standard JPEG quantization matrix based on quality factor (1-100)."""
    quality = max(1, min(100, quality))
    if quality < 50:
        scale = 50.0 / quality
    else:
        scale = (100.0 - quality) / 50.0

    q_table = np.floor(JPEG_LUMA_QUANT * scale + 0.5)
    q_table[q_table < 1] = 1
    q_table[q_table > 255] = 255
    return q_table


def zigzag_indices(n: int = 8) -> list[tuple[int, int]]:
    """Generates the sequence of (row, col) coordinates in zigzag scan order."""
    coords = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # Move up-right
            r = min(s, n - 1)
            c = s - r
            while r >= 0 and c < n:
                coords.append((r, c))
                r -= 1
                c += 1
        else:
            # Move down-left
            c = min(s, n - 1)
            r = s - c
            while c >= 0 and r < n:
                coords.append((r, c))
                r += 1
                c -= 1
    return coords


def zigzag_scan(block: np.ndarray) -> np.ndarray:
    """Scans an 8x8 block into a 1D 64-element array in zigzag order."""
    coords = zigzag_indices(block.shape[0])
    return np.array([block[r, c] for r, c in coords], dtype=block.dtype)


def inverse_zigzag(flat_array: np.ndarray, n: int = 8) -> np.ndarray:
    """Reconstructs an N x N block from a 1D zigzag scanned array."""
    block = np.zeros((n, n), dtype=flat_array.dtype)
    coords = zigzag_indices(n)
    for idx, (r, c) in enumerate(coords):
        block[r, c] = flat_array[idx]
    return block


def compress_image_dct(
    image: np.ndarray, quality: int = 50, block_size: int = 8
) -> tuple[np.ndarray, dict]:
    """Applies block 2D-DCT, quantization, and optional thresholding to an image.

    Args:
        image: 2D uint8 or float grayscale image array.
        quality: Quality factor (1-100).
        block_size: Size of square sub-blocks (default 8).

    Returns:
        reconstructed: Reconstructed image after quantization and IDCT.
        metrics: Dictionary containing MSE, PSNR, and zero-coefficient ratio.
    """
    img_float = image.astype(np.float64) - 128.0  # Zero-center pixels
    h, w = img_float.shape

    # Pad image to multiples of block_size
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    padded = np.pad(img_float, ((0, pad_h), (0, pad_w)), mode="edge")
    padded_h, padded_w = padded.shape

    T = generate_dct_matrix(block_size)
    TT = T.T
    Q = get_scaled_quant_matrix(quality)

    reconstructed_padded = np.zeros_like(padded)
    total_coeffs = 0
    zero_coeffs = 0

    for r in range(0, padded_h, block_size):
        for c in range(0, padded_w, block_size):
            block = padded[r : r + block_size, c : c + block_size]

            # Forward 2D-DCT: D = T * B * T'
            dct_block = T @ block @ TT

            # Quantization: D_q = round(D / Q)
            quantized = np.round(dct_block / Q)

            total_coeffs += block_size * block_size
            zero_coeffs += np.sum(quantized == 0)

            # Dequantization: D_hat = D_q * Q
            dequantized = quantized * Q

            # Inverse 2D-DCT: B_hat = T' * D_hat * T
            idct_block = TT @ dequantized @ T

            reconstructed_padded[r : r + block_size, c : c + block_size] = idct_block

    # Unpad and shift back
    reconstructed = reconstructed_padded[:h, :w] + 128.0
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    mse = compute_mse(image, reconstructed)
    psnr = compute_psnr(image, reconstructed)
    zero_ratio = (zero_coeffs / total_coeffs) * 100.0

    metrics = {
        "mse": float(mse),
        "psnr_db": float(psnr),
        "zero_coefficient_percentage": float(zero_ratio),
        "quality": quality,
    }
    return reconstructed, metrics


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Mean Squared Error between two images."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in decibels."""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float("inf")
    return float(10.0 * math.log10((255.0**2) / mse))
