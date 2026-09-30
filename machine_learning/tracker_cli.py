#!/usr/bin/env python3
"""Unified Adaptive GMM Background Subtraction CLI.

Course: Machine Learning & Computer Vision
Algorithm: Stauffer-Grimson (CVPR 1999 / PAMI 2000)
Author: Sudhanshu Chaudhary (2019JTM2207)
Institution: IIT Delhi
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from config import GMMConfig
from gmm_model import GaussianMixtureBackground
from synthetic_feed import generate_synthetic_video


def run_tracker(
    input_source: str, params_file: str, max_frames: int, output_dir: str | None = None
):
    config = GMMConfig.from_params_file(params_file)
    print("========================================================")
    print(" Adaptive GMM Background Subtraction (Stauffer-Grimson) ")
    print("========================================================")
    print(f"Number of Gaussians (K) : {config.k_gaussians}")
    print(f"Background Threshold (T): {config.threshold}")
    print(f"Learning Rate (alpha)   : {config.learning_rate}")
    print(f"Initial Variance        : {config.initial_variance}")
    print("========================================================")

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if input_source == "synthetic":
        print("[Tracker] Processing synthetic test video stream...")
        height, width = 120, 160
        model = GaussianMixtureBackground(height, width, config)

        start_time = time.time()
        processed = 0
        total_fg_pixels = 0

        for frame_idx, frame, (x1, y1, x2, y2) in generate_synthetic_video(
            num_frames=max_frames, height=height, width=width
        ):
            fg_mask, _bg_model = model.process_frame(frame)
            processed += 1
            fg_count = int(np.sum(fg_mask > 0))
            total_fg_pixels += fg_count

            if frame_idx % 10 == 0:
                print(
                    f"Frame #{frame_idx:03d}: Foreground pixels: {fg_count:4d} (GT Box: [{x1},{y1} to {x2},{y2}])"
                )

            if output_dir and frame_idx % 10 == 0:
                cv2.imwrite(f"{output_dir}/frame_{frame_idx:03d}.png", frame)
                cv2.imwrite(f"{output_dir}/mask_{frame_idx:03d}.png", fg_mask)

        elapsed = time.time() - start_time
        fps = processed / max(1e-4, elapsed)
        print("--------------------------------------------------------")
        print(f"Completed {processed} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
        print(f"Average Foreground Detection: {total_fg_pixels / processed:.1f} pixels/frame")
        print("--------------------------------------------------------")

    else:
        # Video file or camera device
        source = int(input_source) if input_source.isdigit() else input_source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Unable to open video source: {input_source}", file=sys.stderr)
            sys.exit(1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Tracker] Opened video: {input_source} ({width}x{height})")

        model = GaussianMixtureBackground(height, width, config)
        processed = 0
        start_time = time.time()

        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask, _bg_model = model.process_frame(gray)
            processed += 1

            if processed % 50 == 0:
                print(f"Processed {processed} frames...")

        cap.release()
        elapsed = time.time() - start_time
        print(
            f"Processed {processed} frames in {elapsed:.2f}s ({(processed / max(1e-4, elapsed)):.1f} FPS)"
        )


def main():
    parser = argparse.ArgumentParser(description="Adaptive GMM Background Subtraction CLI")
    parser.add_argument(
        "--input",
        type=str,
        default="synthetic",
        help="Input video file, camera index (0), or 'synthetic'",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=str(SCRIPT_DIR / "Params.txt"),
        help="Path to Params.txt configuration file",
    )
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to process")
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save output frame masks"
    )

    args = parser.parse_args()
    run_tracker(args.input, args.params, args.frames, args.output)


if __name__ == "__main__":
    main()
