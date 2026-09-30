"""Synthetic Video Generator for Headless Testing and Verification.

Course: Machine Learning & Computer Vision
"""

from collections.abc import Iterator

import numpy as np


def generate_synthetic_video(
    num_frames: int = 50,
    height: int = 120,
    width: int = 160,
    noise_std: float = 3.0,
    box_size: int = 24,
    speed: int = 3,
) -> Iterator[tuple[int, np.ndarray, tuple[int, int, int, int]]]:
    """Generates synthetic video frames featuring a moving square against a static textured background.

    Yields:
        (frame_index, gray_frame_uint8, ground_truth_bbox (x1, y1, x2, y2))
    """
    np.random.seed(42)
    # Base background: gentle diagonal gradient + static texture
    y_coords, x_coords = np.ogrid[:height, :width]
    base_bg = 60.0 + 40.0 * (x_coords / width) + 20.0 * (y_coords / height)

    x_pos = 10
    y_pos = height // 2 - box_size // 2

    for frame_idx in range(num_frames):
        frame = base_bg.copy()

        # Add Gaussian sensor noise
        if noise_std > 0:
            frame += np.random.normal(0, noise_std, (height, width))

        # Only insert foreground moving object after frame 10 (giving model time to learn background)
        bbox = (0, 0, 0, 0)
        if frame_idx >= 10:
            x1 = int(x_pos)
            y1 = int(y_pos)
            x2 = min(width, x1 + box_size)
            y2 = min(height, y1 + box_size)
            # High intensity moving object
            frame[y1:y2, x1:x2] = 230.0
            bbox = (x1, y1, x2, y2)
            x_pos = (x_pos + speed) % (width - box_size)

        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        yield frame_idx, frame_uint8, bbox
