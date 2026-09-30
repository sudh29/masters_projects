#!/usr/bin/env python3
"""Unified Audio Compression CLI (DPCM & Golomb-Rice).

Course: ELL-786 Multimedia Systems (Assignment 3)
Author: Sudhanshu Chaudhary (2019JTM2207)
Institution: IIT Delhi
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from audio_pipeline import load_wav, process_dpcm_pipeline, save_wav


def main():
    parser = argparse.ArgumentParser(description="IIT Delhi ELL-786 Audio DPCM Compression Suite")
    parser.add_argument(
        "--input",
        type=str,
        default=str(SCRIPT_DIR / "1Dialogue.wav"),
        help="Input WAV audio file",
    )
    parser.add_argument(
        "--order", type=int, default=2, choices=[1, 2, 3, 4], help="Predictor filter order (1-4)"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[2, 4, 6, 8, 10, 12, 14, 16],
        help="Quantization bit depth",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for reconstructed WAV audio"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run multi-order comparison table (N=1, 2, 4)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Audio file not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    samples, sample_rate = load_wav(input_path)
    print("\n========================================================")
    print(f" Loaded Audio: {input_path.name}")
    print(f" Samples: {len(samples)} | Sample Rate: {sample_rate} Hz | Bit Depth: 16-bit")
    print("========================================================")

    if args.compare:
        print("\nRunning Multi-Order Prediction Comparison...")
        print("-------------------------------------------------------------------------")
        print(" Order (N) | Bits | Step Size |   SNR (dB)  |  SPER (dB)  | Bits/Sample")
        print("-------------------------------------------------------------------------")
        for order in [1, 2, 4]:
            for bits in [4, 8, 12]:
                res = process_dpcm_pipeline(samples, order=order, bits=bits)
                print(
                    f"    {res['order']}      |  {res['bits']:2d}  |  {res['delta']:8.2f} |  {res['snr_db']:7.2f} dB | {res['sper_db']:7.2f} dB | {res['avg_bits_per_sample']:6.2f} b"
                )
        print("-------------------------------------------------------------------------")
    else:
        res = process_dpcm_pipeline(samples, order=args.order, bits=args.bits)
        print(f"Predictor Order (N)           : {res['order']}")
        print(f"Predictor Coefficients (A)    : {[round(c, 4) for c in res['coefficients']]}")
        print(f"Quantizer Bit Depth           : {res['bits']} bits")
        print(f"Quantization Step Size (Delta): {res['delta']:.4f}")
        print(f"Golomb Parameter (M)          : {res['m_param']}")
        print(f"Signal-to-Noise Ratio (SNR)   : {res['snr_db']:.2f} dB")
        print(f"Signal-to-Pred-Error (SPER)   : {res['sper_db']:.2f} dB")
        print(f"Average Bitrate               : {res['avg_bits_per_sample']:.2f} bits/sample")
        print(f"Estimated Compression Ratio   : {res['compression_ratio']:.2f}x")
        print("--------------------------------------------------------")

        if args.output:
            save_wav(args.output, res["reconstructed"], sample_rate)
            print(f"Reconstructed audio saved to: {args.output}")


if __name__ == "__main__":
    main()
