"""Regenerate the fixed 128³ synthetic reference used by PAPER.md."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import tifffile
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.build import load_generator
from src.train.weights import find_weights

REFERENCE_SIZE = 128
SEED = 10_000
OUTPUT = PROJECT_ROOT / "scripts" / "gt_128.tiff"
MANIFEST = PROJECT_ROOT / "temp" / "paper_reference_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight",
        type=Path,
        help="generator weight (default: newest run/*/generator.pt)",
    )
    args = parser.parse_args()

    weights = (
        args.weight.resolve()
        if args.weight is not None
        else find_weights(PROJECT_ROOT / "run").resolve()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(weights, device=device)
    if generator.patch_size != REFERENCE_SIZE:
        raise ValueError(
            f"paper reference requires patch size {REFERENCE_SIZE}, "
            f"got {generator.patch_size}."
        )

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    volume = generator.generate().to(torch.uint8).numpy()
    payload = volume.tobytes()
    digest = hashlib.sha256(payload).hexdigest()

    temporary = OUTPUT.with_name(f".{OUTPUT.stem}.tmp{OUTPUT.suffix}")
    try:
        tifffile.imwrite(temporary, volume)
        temporary.replace(OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "weights": str(weights),
                "seed": SEED,
                "shape": list(volume.shape),
                "dtype": str(volume.dtype),
                "sha256_voxels": digest,
                "phase_fractions": [
                    float((volume == phase).mean())
                    for phase in range(generator.num_phases)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Weights  : {weights}")
    print(f"Reference: {OUTPUT.resolve()}")
    print(f"Manifest : {MANIFEST.resolve()}")
    print(f"SHA-256  : {digest}")


if __name__ == "__main__":
    main()
