import argparse
import hashlib
import json
import sys
from pathlib import Path

import tifffile
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from provenance import (
    build_provenance,
    file_record,
    validate_output_paths,
)

from src.build import load_generator
from src.config import load_generation_settings

REFERENCE_SIZE = 128
SEED = 10_000
OUTPUT = PROJECT_ROOT / "scripts" / "gt_128.tiff"
MANIFEST = PROJECT_ROOT / "temp" / "paper_reference_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
    )
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float)
    args = parser.parse_args()

    weights = args.weight.resolve()
    settings = load_generation_settings(weights)
    guidance_scale = (
        settings.guidance_scale if args.guidance_scale is None else args.guidance_scale
    )
    provenance = build_provenance(
        weights,
        guidance_scale,
        generation={
            "seed": SEED,
            "domain": args.domain,
            "output_size": REFERENCE_SIZE,
        },
    )
    validate_output_paths(provenance, (OUTPUT, MANIFEST))
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
    volume = (
        generator.generate(
            guidance_scale=guidance_scale,
            domain=args.domain,
        )
        .to(torch.uint8)
        .numpy()
    )
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
                **provenance,
                "seed": SEED,
                "output": {
                    **file_record(OUTPUT),
                    "shape": list(volume.shape),
                    "dtype": str(volume.dtype),
                },
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
    print(f"Guidance : {guidance_scale}")
    print(f"Reference: {OUTPUT.resolve()}")
    print(f"Manifest : {MANIFEST.resolve()}")
    print(f"SHA-256  : {digest}")


if __name__ == "__main__":
    main()
