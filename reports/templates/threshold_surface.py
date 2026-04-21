from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from jointadaspec.inference import JointAdaSpecPolicy
from reports.templates._common import manifest_footnote


def main() -> int:
    parser = argparse.ArgumentParser(description="Render JointAdaSpec threshold surfaces.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    policy_path = Path(args.policy)
    out_dir = Path(args.out)
    manifest_path = None if args.manifest is None else Path(args.manifest)
    policy = JointAdaSpecPolicy.load(policy_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k_value in range(policy.config.gamma_max):
        length_surface = policy.export_length_surface(k_value)
        threshold_surface = policy.export_threshold_surface(k_value)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        image_left = axes[0].imshow(length_surface, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        axes[0].set_title(f"h*(K, k={k_value})")
        axes[0].set_xlabel("K bin")
        axes[0].set_ylabel("H bin")
        fig.colorbar(image_left, ax=axes[0], fraction=0.046, pad=0.04)

        image_right = axes[1].imshow(threshold_surface, origin="lower", aspect="auto", cmap="magma")
        axes[1].set_title(f"T*(H, K, k={k_value})")
        axes[1].set_xlabel("K bin")
        axes[1].set_ylabel("H bin")
        fig.colorbar(image_right, ax=axes[1], fraction=0.046, pad=0.04)
        fig.text(0.01, 0.01, manifest_footnote(manifest_path, [policy_path]), fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(out_dir / f"threshold_surface_k{k_value}.pdf")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
