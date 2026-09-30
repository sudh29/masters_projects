"""Configuration management for Adaptive GMM Background Subtraction.

Course: Machine Learning & Computer Vision
Algorithm: Stauffer-Grimson Adaptive Background Mixture Model (CVPR 1999 / PAMI 2000)
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GMMConfig:
    k_gaussians: int = 3
    threshold: float = 0.58
    learning_rate: float = 0.1
    initial_mean: float = 130.0
    initial_variance: float = 36.0
    initial_weight: float = 1.0 / 3.0
    match_distance_std: float = 2.5  # Mahalanobis distance multiplier (D < 2.5 * sigma)

    @classmethod
    def from_params_file(cls, filepath: str | Path) -> "GMMConfig":
        """Loads configuration from Params.txt or similar key-value files."""
        path = Path(filepath)
        if not path.exists():
            return cls()

        cfg = cls()
        lines = path.read_text().splitlines()
        for line in lines:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = [p.strip() for p in line.split("=", 1)]
            key_lower = key.lower()

            try:
                if "number of gaussians" in key_lower:
                    cfg.k_gaussians = int(val)
                elif "threshold" in key_lower:
                    cfg.threshold = float(val)
                elif "learningrate" in key_lower or "alpha" in key_lower:
                    cfg.learning_rate = float(val)
                elif "mean" in key_lower:
                    cfg.initial_mean = float(val)
                elif "weight" in key_lower:
                    if "/" in val:
                        num, den = val.split("/")
                        cfg.initial_weight = float(num) / float(den)
                    else:
                        cfg.initial_weight = float(val)
            except ValueError:
                continue

        return cfg
