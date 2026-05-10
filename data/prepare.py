"""Stage: data preparation.

Materializes the bundled Schraivogel gRNA-count h5ad as
`<out>` for downstream stages. Sources, in order of preference:

1. `--src <path>` if given.
2. Sibling checkout `../crispat/example_data/Schraivogel/gRNA_counts.h5ad`
   (laptop case, where the crispat repo is alongside crispat-profiling).
3. Download from the upstream repo's raw URL at the pinned commit
   (cluster case, where there's no sibling checkout).

The downloaded copy is cached under `<out>` so reruns are no-ops.
"""
import argparse
import shutil
import urllib.request
from pathlib import Path

SIBLING = Path(__file__).resolve().parents[1].parent / "crispat" / "example_data" / "Schraivogel" / "gRNA_counts.h5ad"
PINNED_COMMIT = "8d96f10"
RAW_URL = f"https://raw.githubusercontent.com/velten-group/crispat/{PINNED_COMMIT}/example_data/Schraivogel/gRNA_counts.h5ad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output h5ad path")
    ap.add_argument("--src", default=None, help="Optional explicit source path")
    args = ap.parse_args()

    dst = Path(args.out)
    if dst.exists():
        print(f"data: {dst} already present, skipping")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)

    if args.src:
        src = Path(args.src)
        if not src.exists():
            raise FileNotFoundError(f"--src not found: {src}")
        shutil.copy2(src, dst)
        print(f"data: copied {src} -> {dst}")
        return

    if SIBLING.exists():
        shutil.copy2(SIBLING, dst)
        print(f"data: copied {SIBLING} -> {dst}")
        return

    print(f"data: downloading from {RAW_URL}")
    urllib.request.urlretrieve(RAW_URL, dst)
    print(f"data: saved -> {dst}")


if __name__ == "__main__":
    main()
