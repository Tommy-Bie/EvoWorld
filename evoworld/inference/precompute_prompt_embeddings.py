"""Precompute prompt embeddings for EvoWorld inference."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


_PKG = Path(__file__).resolve().parent.parent / "packages" / "ltx-trainer" / "src"
if _PKG.is_dir():
    sys.path.insert(0, str(_PKG))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True, help="benchmark data/testset.jsonl")
    ap.add_argument("--out-dir", required=True, help="Embedding output directory")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N rows")
    ap.add_argument("--fresh", action="store_true", help="Recompute existing files")
    a = ap.parse_args()

    for k in ("EW_GEMMA", "EW_CKPT"):
        if not os.environ.get(k):
            return int(bool(print(f"Missing required environment variable: {k}")))

    from ltx_trainer.model_loader import load_text_encoder, load_embeddings_processor, embedding_weight_paths
    E = os.environ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = [json.loads(l) for l in open(a.testset, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    todo = []
    for r in rows:
        p = out_dir / (r["clip_id"].replace("/", "__") + ".pt")
        if p.is_file() and not a.fresh:
            continue
        todo.append((r, p))
    print(f"[precompute] rows={len(rows)} pending={len(todo)} skipped={len(rows)-len(todo)}",
          flush=True)
    if not todo:
        print("[precompute] Nothing to do.")
        return 0


    print("[precompute] Loading the text encoder and embeddings processor...", flush=True)
    te = load_text_encoder(gemma_model_path=E["EW_GEMMA"], device=device, dtype=torch.bfloat16, load_in_8bit=False)
    ep = load_embeddings_processor(
        checkpoint_path=embedding_weight_paths(E["EW_CKPT"], E["EW_GEMMA"]),
        gemma_model_path=E["EW_GEMMA"], device=device, dtype=torch.bfloat16)

    @torch.no_grad()
    def enc(prompt_text):
        hs, mask = te.encode([prompt_text])[0]
        out = ep.process_hidden_states(hs, mask)
        return {"video": out.video_encoding.cpu(),
                "audio": (out.audio_encoding.cpu() if out.audio_encoding is not None else None)}


    neg = enc("")

    ok = 0
    t0 = time.time()
    for i, (r, p) in enumerate(todo, 1):
        rec = {"clip_id": r["clip_id"], "neg": neg}
        for seg, key in (("seg0", "initial_prompt"), ("seg1", "event_prompt"), ("seg2", "gt_revisit_prompt")):
            txt = (r.get(key) or "").strip()
            rec[seg] = enc(txt) if txt else None
        torch.save(rec, p)
        ok += 1
        if i % 20 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {r['clip_id']} ({(time.time()-t0)/i:.1f}s/item)", flush=True)

    print(f"\n[precompute] Wrote {ok} items to {out_dir}")
    print("  Reuse them with deploy_pipeline.py --prompt-embeds-dir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
