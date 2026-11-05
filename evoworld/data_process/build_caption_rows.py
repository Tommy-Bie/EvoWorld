"""Build segment-level caption rows for preprocessing."""
import argparse, json, collections
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)


    ap.add_argument("--with-global", action="store_true", default=True)
    a = ap.parse_args()

    rows, per_nseg, n_clip = [], collections.Counter(), 0
    for line in open(a.captions, encoding="utf-8"):
        r = json.loads(line)
        sp = r.get("segment_prompts")
        if not sp:
            continue
        n_clip += 1
        per_nseg[len(sp)] += 1
        for i, c in enumerate(sp):
            rows.append({"media_path": f"{r['clip_id']}__seg{i}.mp4", "caption": c})
        if a.with_global and r.get("global_prompt"):
            rows.append({"media_path": f"{r['clip_id']}__global.mp4",
                         "caption": r["global_prompt"]})

    dup = [p for p, n in collections.Counter(x["media_path"] for x in rows).items() if n > 1]
    if dup:
        raise SystemExit(f"Duplicate media paths: {len(dup)}. Examples: {dup[:3]}")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{n_clip} clips, segment distribution {dict(per_nseg)} -> {len(rows)} rows")
    print(f"  Wrote {out}")
    print("  Duplicate media paths: 0")


if __name__ == "__main__":
    main()
