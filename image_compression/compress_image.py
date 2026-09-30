#!/usr/bin/env python3
"""Unified Image & Data Compression CLI.

Course: ELL-786 Multimedia Systems
Author: Sudhanshu Chaudhary (2019JTM2207)
Institution: IIT Delhi

Supported Algorithms:
  - dct         : 2D Block Discrete Cosine Transform with JPEG Quantization
  - arithmetic  : Exact Interval Arithmetic Coding
  - lz77        : Sliding-window LZ77 compression
  - lz78        : Tree-based LZ78 compression
  - lzw         : Lempel-Ziv-Welch compression
  - repetition  : Repetition Channel Coding with Error Correction
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

# Ensure src is importable
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from arithmetic_codec import decode_arithmetic, encode_arithmetic
from dct_codec import compress_image_dct
from dictionary_codecs import (
    lz77_decode,
    lz77_encode,
    lz78_decode,
    lz78_encode,
    lzw_decode,
    lzw_encode,
)
from repetition_codec import (
    inject_hamming_errors,
    repetition_decode,
    repetition_encode,
    str_to_bits,
)


def run_dct(input_path: str, quality: int, output_path: str | None = None):
    print(f"\n[DCT Compression] Loading image: {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image at {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Original Resolution : {img.shape[1]}x{img.shape[0]} (Grayscale)")
    print(f"Quality Factor      : {quality}")

    reconstructed, metrics = compress_image_dct(img, quality=quality)

    print("--------------------------------------------------")
    print(f"MSE (Mean Squared Error)       : {metrics['mse']:.4f}")
    print(f"PSNR (Peak Signal-to-Noise)    : {metrics['psnr_db']:.2f} dB")
    print(f"Zero Coefficients (Energy Drop): {metrics['zero_coefficient_percentage']:.2f}%")
    print("--------------------------------------------------")

    if output_path:
        cv2.imwrite(output_path, reconstructed)
        print(f"Reconstructed image saved to: {output_path}")


def run_text_codec(algo: str, text: str):
    print(f'\n[{algo.upper()} Compression] Input text: "{text}" ({len(text)} chars)')

    if algo == "lz77":
        tokens = lz77_encode(text)
        decoded = lz77_decode(tokens)
        print(f"LZ77 Tokens ({len(tokens)}): {tokens[:8]}...")
        assert decoded == text
        print(f'Lossless Verification: PASSED (Reconstructed: "{decoded}")')

    elif algo == "lz78":
        tokens = lz78_encode(text)
        decoded = lz78_decode(tokens)
        print(f"LZ78 Tokens ({len(tokens)}): {tokens[:8]}...")
        assert decoded == text
        print(f'Lossless Verification: PASSED (Reconstructed: "{decoded}")')

    elif algo == "lzw":
        codes = lzw_encode(text)
        decoded = lzw_decode(codes)
        print(f"LZW Codes ({len(codes)}): {codes[:10]}...")
        assert decoded == text
        print(f"Compression Ratio    : {len(text) / max(1, len(codes)):.2f}x")
        print(f'Lossless Verification: PASSED (Reconstructed: "{decoded}")')

    elif algo == "arithmetic":
        tag, length, intervals = encode_arithmetic(text)
        decoded = decode_arithmetic(tag, length, intervals)
        print(f"Arithmetic Tag       : {float(tag):.12f} ({tag})")
        assert decoded == text
        print(f'Lossless Verification: PASSED (Reconstructed: "{decoded}")')

    elif algo == "repetition":
        bits = str_to_bits(text)
        encoded = repetition_encode(bits, r=3)
        corrupted = inject_hamming_errors(encoded, error_count=2, seed=42)
        recovered_bits = repetition_decode(corrupted, r=3)
        print(f"Original Bits ({len(bits)}) : {bits[:24]}...")
        print(f"Encoded (3,1) Repetition   : {encoded[:24]}...")
        print(f"Corrupted with 2 Errors     : {corrupted[:24]}...")
        print(f"Majority Corrected Bits     : {recovered_bits[:24]}...")
        assert recovered_bits == bits
        print("Error Correction Capability: PASSED (100% recovered despite channel errors)")


def main():
    parser = argparse.ArgumentParser(description="IIT Delhi ELL-786 Multimedia Compression Suite")
    parser.add_argument(
        "--algo",
        choices=["dct", "arithmetic", "lz77", "lz78", "lzw", "repetition"],
        default="dct",
        help="Compression algorithm to execute",
    )
    parser.add_argument("--input", type=str, default=None, help="Input file path (image or text)")
    parser.add_argument(
        "--text",
        type=str,
        default="a.bar.array.by.barrayar.bay.",
        help="Sample text for text codecs",
    )
    parser.add_argument(
        "--quality", type=int, default=50, help="JPEG quality factor (1-100) for DCT"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for reconstructed image"
    )

    args = parser.parse_args()

    if args.algo == "dct":
        img_path = args.input or str(SCRIPT_DIR / "cat.jpg")
        run_dct(img_path, quality=args.quality, output_path=args.output)
    else:
        text = args.text
        if args.input and os.path.exists(args.input):
            with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        run_text_codec(args.algo, text)


if __name__ == "__main__":
    main()
