"""Create reference-latent caches for EvoWorld training."""
import argparse, os, sys
from pathlib import Path

import torch

REF_LAT = 2


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--latents", default=os.environ.get("EW_LATENTS"),
                    required="EW_LATENTS" not in os.environ,
                    help="Directory containing full-video latent tensors")
    ap.add_argument("--out", default=os.environ.get("EW_REFERENCE"),
                    required="EW_REFERENCE" not in os.environ,
                    help="Output directory matching the reference latents_dir in the config")
    ap.add_argument("--ref-lat", type=int, default=REF_LAT)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    src, dst = Path(a.latents), Path(a.out)
    files = sorted(src.rglob("*.pt"))
    if not files:
        sys.exit(f"No .pt files found under {src}")
    print(f"{len(files)} full-video tensors -> {a.ref_lat} reference latents")

    done = skip = bad = 0
    shapes = {}
    for i, f in enumerate(files, 1):
        rel = f.relative_to(src)
        out = dst / rel
        if out.exists() and not a.overwrite:
            skip += 1
            continue
        d = torch.load(f, map_location="cpu", weights_only=True)
        lat = d["latents"]
        if lat.dim() != 4:
            print(f"  Invalid latent shape for {rel}: {tuple(lat.shape)}; expected [C,F,H,W]")
            bad += 1
            continue
        if lat.shape[1] < a.ref_lat:
            print(f"  {rel} has {lat.shape[1]} latents; expected at least {a.ref_lat}")
            bad += 1
            continue
        ref = lat[:, : a.ref_lat].contiguous()
        rec = {
            "latents": ref,


            "num_frames": int(a.ref_lat),
            "height": int(d["height"]),
            "width": int(d["width"]),


            "fps": d["fps"],
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".pt.tmp")
        torch.save(rec, tmp)
        tmp.replace(out)
        shapes[tuple(ref.shape)] = shapes.get(tuple(ref.shape), 0) + 1
        done += 1
        if i % 1000 == 0:
            print(f"  {i}/{len(files)} written={done} skipped={skip} invalid={bad}", flush=True)

    print(f"\nDone: written={done} skipped={skip} invalid={bad}")
    for s, n in sorted(shapes.items()):
        print(f"  shape {s}: {n}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
