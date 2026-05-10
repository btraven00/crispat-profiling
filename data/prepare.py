"""Stage: data preparation.

Copies the bundled Schraivogel gRNA-count h5ad into the results tree as the
deterministic input for downstream stages.
"""
import argparse
import shutil
from pathlib import Path

BUNDLED = Path(__file__).resolve().parents[1].parent / "crispat" / "example_data" / "Schraivogel" / "gRNA_counts.h5ad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output h5ad path")
    args = ap.parse_args()

    src = BUNDLED
    if not src.exists():
        raise FileNotFoundError(f"Bundled dataset not found at {src}")

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"data: copied {src} -> {dst}")


if __name__ == "__main__":
    main()
