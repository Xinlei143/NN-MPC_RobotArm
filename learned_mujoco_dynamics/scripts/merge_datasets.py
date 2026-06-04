from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_dynamics.dataset_merge import merge_npz_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge MuJoCo dynamics .npz datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inputs", nargs="+", required=True, type=Path, help="Input .npz dataset paths")
    parser.add_argument("--output", required=True, type=Path, help="Output merged .npz path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shapes = merge_npz_datasets(args.inputs, args.output)
    print(f"Saved merged dataset to {args.output}")
    for name, shape in shapes.items():
        print(f"{name}={shape}")


if __name__ == "__main__":
    main()
