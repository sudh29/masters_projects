"""Adaptive Gaussian Mixture Model (GMM) Background Subtraction.

Course: Machine Learning & Computer Vision
Algorithm: Stauffer-Grimson (CVPR 1999 / PAMI 2000)
Author: Sudhanshu Chaudhary (2019JTM2207)
"""

import numpy as np
from config import GMMConfig


class GaussianMixtureBackground:
    """Adaptive Background Subtractor using Mixture of Gaussians (MoG / GMM)."""

    def __init__(self, height: int, width: int, config: GMMConfig = None):
        self.height = height
        self.width = width
        self.config = config or GMMConfig()
        self.k = self.config.k_gaussians

        # State tensors: shape (H, W, K)
        self.mu = np.zeros((height, width, self.k), dtype=np.float64)
        self.var = np.zeros((height, width, self.k), dtype=np.float64)
        self.wt = np.zeros((height, width, self.k), dtype=np.float64)

        self.initialized = False
        self.reset()

    def reset(self):
        """Initializes or resets all Gaussian parameters across the frame."""
        self.mu.fill(self.config.initial_mean)
        self.var.fill(self.config.initial_variance)
        self.wt.fill(1.0 / self.k)
        self.initialized = False

    def initialize_with_frame(self, gray_frame: np.ndarray):
        """Initializes the dominant Gaussian mean directly from the first observed frame."""
        frame = gray_frame.astype(np.float64)
        self.mu[:, :, 0] = frame
        for k in range(1, self.k):
            self.mu[:, :, k] = np.clip(frame + (k * 10.0 - 5.0), 0, 255)
        self.var.fill(self.config.initial_variance)
        self.wt.fill(1.0 / self.k)
        self.initialized = True

    def process_frame(self, gray_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Processes a single grayscale video frame and updates background models.

        Args:
            gray_frame: 2D uint8 grayscale image (H, W).

        Returns:
            fg_mask: Binary mask where 255 represents foreground and 0 represents background.
            bg_image: Reconstructed background visual model based on dominant Gaussian means.
        """
        frame = gray_frame.astype(np.float64)
        if not self.initialized:
            self.initialize_with_frame(gray_frame)

        h, w = self.height, self.width
        alpha = self.config.learning_rate
        mult = self.config.match_distance_std

        # Standard deviations: shape (H, W, K)
        std = np.sqrt(np.maximum(self.var, 1e-4))

        # Check matching condition for all K Gaussians: |I - mu| < 2.5 * sigma
        dist = np.abs(frame[:, :, np.newaxis] - self.mu)
        match_mask = dist < (mult * std)  # shape (H, W, K) boolean

        # Find first matching Gaussian index per pixel (-1 if none matched)
        matched_idx = np.full((h, w), -1, dtype=np.int32)
        for k in range(self.k):
            is_first_match = match_mask[:, :, k] & (matched_idx == -1)
            matched_idx[is_first_match] = k

        has_match = matched_idx != -1

        # ----------------------------------------------------------------------
        # 1. Update weights for all pixels
        # ----------------------------------------------------------------------
        self.wt = (1.0 - alpha) * self.wt
        for k in range(self.k):
            matched_here = has_match & (matched_idx == k)
            self.wt[matched_here, k] += alpha

        # ----------------------------------------------------------------------
        # 2. Update mean and variance for matched components
        # ----------------------------------------------------------------------
        for k in range(self.k):
            matched_here = has_match & (matched_idx == k)
            if not np.any(matched_here):
                continue

            diff = frame[matched_here] - self.mu[matched_here, k]
            sigma_k = std[matched_here, k]

            # Gaussian probability density
            gaussian_pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sigma_k)) * np.exp(
                -0.5 * (diff / sigma_k) ** 2
            )
            rho = np.clip(alpha * gaussian_pdf, 1e-4, 1.0)

            # Update parameters
            self.mu[matched_here, k] = (1.0 - rho) * self.mu[matched_here, k] + rho * frame[
                matched_here
            ]
            self.var[matched_here, k] = (1.0 - rho) * self.var[matched_here, k] + rho * (diff**2)
            self.var[matched_here, k] = np.maximum(self.var[matched_here, k], 4.0)  # Bound variance

        # ----------------------------------------------------------------------
        # 3. Handle unmatched pixels (replace least fit Gaussian)
        # ----------------------------------------------------------------------
        no_match = ~has_match
        if np.any(no_match):
            # Compute fitness ratio omega / sigma
            fitness = self.wt / std
            worst_k = np.argmin(fitness, axis=2)

            for k in range(self.k):
                replace_here = no_match & (worst_k == k)
                if np.any(replace_here):
                    self.mu[replace_here, k] = frame[replace_here]
                    self.var[replace_here, k] = self.config.initial_variance
                    self.wt[replace_here, k] = 0.05  # Low initial weight

        # Normalize weights so sum_k(wt_k) == 1
        wt_sum = np.sum(self.wt, axis=2, keepdims=True)
        self.wt = self.wt / np.maximum(wt_sum, 1e-6)

        # ----------------------------------------------------------------------
        # 4. Sort Gaussians by fitness ratio (omega / sigma) descending
        # ----------------------------------------------------------------------
        fitness = self.wt / np.sqrt(np.maximum(self.var, 1e-4))
        sort_order = np.argsort(-fitness, axis=2)

        # Apply sort order to mu, var, wt
        self.mu = np.take_along_axis(self.mu, sort_order, axis=2)
        self.var = np.take_along_axis(self.var, sort_order, axis=2)
        self.wt = np.take_along_axis(self.wt, sort_order, axis=2)

        # ----------------------------------------------------------------------
        # 5. Background Model Classification
        # ----------------------------------------------------------------------
        # Find threshold B: first B Gaussians with cumulative weight > threshold
        cum_weights = np.cumsum(self.wt, axis=2)
        bg_rank = np.zeros((h, w), dtype=np.int32)
        for k in range(self.k):
            exceeds = (cum_weights[:, :, k] >= self.config.threshold) & (bg_rank == 0)
            bg_rank[exceeds] = k

        # Check rank of matching Gaussian under the newly sorted ordering
        # Re-check match against top background Gaussians
        fg_mask = np.full((h, w), 255, dtype=np.uint8)
        for k in range(self.k):
            in_bg_model = k <= bg_rank
            d_k = np.abs(frame - self.mu[:, :, k])
            is_match = d_k < (mult * np.sqrt(self.var[:, :, k]))
            # If matches a background model distribution, classify as background (0)
            is_bg = in_bg_model & is_match
            fg_mask[is_bg] = 0

        # Background visual representation: mean of rank 0 Gaussian
        bg_image = np.clip(self.mu[:, :, 0], 0, 255).astype(np.uint8)

        return fg_mask, bg_image
