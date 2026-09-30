"""Automated test suite for Machine Learning GMM Background Subtraction.

Tests:
1. Configuration parser validation against Params.txt.
2. GMM state tensor initialization and weight sum constraints.
3. Background model adaptation and convergence on static feeds.
4. Foreground segmentation sensitivity upon moving object injection.
5. End-to-end synthetic video tracker stream execution.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = REPO_ROOT / "machine_learning"

sys.path.insert(0, str(ML_DIR / "src"))

from config import GMMConfig
from gmm_model import GaussianMixtureBackground
from synthetic_feed import generate_synthetic_video


def test_config_from_params_file():
    params_path = ML_DIR / "Params.txt"
    assert params_path.exists()
    cfg = GMMConfig.from_params_file(params_path)
    assert cfg.k_gaussians == 3
    assert abs(cfg.threshold - 0.58) < 1e-4
    assert abs(cfg.learning_rate - 0.1) < 1e-4
    assert abs(cfg.initial_mean - 130.0) < 1e-4
    assert abs(cfg.initial_weight - 1.0 / 3.0) < 1e-4


def test_gmm_initialization():
    h, w = 40, 60
    model = GaussianMixtureBackground(h, w, GMMConfig(k_gaussians=4))
    assert model.mu.shape == (h, w, 4)
    assert model.var.shape == (h, w, 4)
    assert model.wt.shape == (h, w, 4)

    # Weights must sum to 1.0 per pixel
    wt_sum = np.sum(model.wt, axis=2)
    np.testing.assert_allclose(wt_sum, np.ones((h, w)), atol=1e-5)


def test_static_background_adaptation():
    """Verify that constant frames are classified as background with 0 foreground."""
    h, w = 30, 40
    model = GaussianMixtureBackground(h, w, GMMConfig(learning_rate=0.2))

    constant_frame = np.full((h, w), 100, dtype=np.uint8)

    # Process 5 identical frames
    for _ in range(5):
        fg_mask, bg_image = model.process_frame(constant_frame)

    # After initial stabilization, all pixels should be classified as background (0)
    assert np.all(fg_mask == 0)
    # The dominant Gaussian mean should converge toward 100
    assert np.all(np.abs(bg_image.astype(int) - 100) < 5)


def test_moving_object_detection():
    """Verify that a sudden novel patch is segmented as foreground."""
    h, w = 50, 50
    model = GaussianMixtureBackground(h, w, GMMConfig(learning_rate=0.1))

    # Train on static dark background
    bg_frame = np.full((h, w), 50, dtype=np.uint8)
    for _ in range(10):
        model.process_frame(bg_frame)

    # Introduce a bright 10x10 patch (intensity 220)
    test_frame = bg_frame.copy()
    test_frame[20:30, 20:30] = 220

    fg_mask, _ = model.process_frame(test_frame)

    # The patch pixels must be identified as foreground (255)
    patch_fg = fg_mask[20:30, 20:30]
    assert np.all(patch_fg == 255)

    # The unperturbed background should remain background (0)
    assert np.sum(fg_mask[0:10, 0:10]) == 0


def test_synthetic_video_stream():
    """Verify processing through the synthetic video feed."""
    h, w = 60, 80
    model = GaussianMixtureBackground(h, w)

    fg_counts = []
    for frame_idx, frame, bbox in generate_synthetic_video(num_frames=20, height=h, width=w):
        fg_mask, _ = model.process_frame(frame)
        fg_counts.append(int(np.sum(fg_mask > 0)))

    # In frames 10-19 an object moves, so foreground detections must occur
    assert sum(fg_counts[10:]) > 0
