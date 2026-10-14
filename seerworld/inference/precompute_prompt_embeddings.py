"""预编码 benchmark testset 的三段 prompt embedding，存盘供推理复用（省掉每 clip 重加载 gemma）。

背景：deploy_pipeline / latent_rollout 每个 clip 都重建 ValidationRunner，其 init 会
【加载 12B gemma → 编码 prompt → 卸载】。150 clip = 加载 150 次 gemma，纯浪费。预编码一次存盘后，
推理端直接 load embedding、完全不碰 gemma，后续换 seed / 换 LoRA step 反复重推都飞快。

★ 三段【分开存】（用户要求）：
  - seg0 = initial_prompt   （i2v/v2v 都用）
  - seg1 = event_prompt     （i2v/v2v 都用）
  - seg2 = gt_revisit_prompt（仅 GT-seg2 模式用；VLM 脑模式 seg2 是现场生成的，无法预编码）
  这样 VLM 推理模式只 load seg0/seg1，GT 模式 load 三段。negative（空串）所有 clip 共享，只存一份。

编码逻辑与 deploy_pipeline.make_prompt_encoder / ValidationRunner._cache_prompt_embeddings
【完全一致】（同一个 gemma + embeddings_processor + 空串 negative），数值等价，不影响生成。

存盘格式：<out_dir>/<clip_safe>.pt = {
    "clip_id": str,
    "neg": {video, audio},                       # 共享的 negative（也各 clip 存一份，load 简单）
    "seg0"/"seg1"/"seg2": {video_pos, audio_pos} 或 None（该段 prompt 为空时）
}
其中 video/audio 张量已 .cpu()。推理端用 rebuild_cached() 拼回 CachedPromptEmbeddings。

用法（配置所需环境变量后，单卡即可）：
  CW_GPU=0 python precompute_prompt_embeddings.py \
      --testset <benchmark>/data/testset.jsonl \
      --out-dir <CW_DATA>/prompt_embeds_bench      # 建议放共享存储，供多机复用
  # 复用：断点续跑（已存在的 .pt 跳过）；--fresh 强制重编。
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# ★ ltx_trainer 在其 packages/ltx-trainer/src 下。deploy/latent_rollout 靠调用方 cd 到
#   packages/ltx-trainer 才能 import；这里插进 sys.path，从哪都能跑（不再依赖调用方 cd）。
_PKG = Path(__file__).resolve().parent.parent / "packages" / "ltx-trainer" / "src"
if _PKG.is_dir():
    sys.path.insert(0, str(_PKG))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True, help="benchmark data/testset.jsonl")
    ap.add_argument("--out-dir", required=True, help="embedding 输出目录（建议共享存储）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试）")
    ap.add_argument("--fresh", action="store_true", help="强制重编，不跳过已存在的 .pt")
    a = ap.parse_args()

    for k in ("CW_GEMMA", "CW_CKPT"):
        if not os.environ.get(k):
            return int(bool(print(f"!! {k} 未设置（请先配置模型路径环境变量）")))

    from ltx_trainer.model_loader import load_text_encoder, load_embeddings_processor, embedding_weight_paths
    E = os.environ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = [json.loads(l) for l in open(a.testset, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 需要编哪些（跳过已存在的）
    todo = []
    for r in rows:
        p = out_dir / (r["clip_id"].replace("/", "__") + ".pt")
        if p.is_file() and not a.fresh:
            continue
        todo.append((r, p))
    print(f"[precompute] testset {len(rows)} 条，待编码 {len(todo)} 条（已存在 {len(rows)-len(todo)} 条跳过）",
          flush=True)
    if not todo:
        print("[precompute] 全部已存在，无需编码。")
        return 0

    # 加载 gemma + embeddings_processor（只加载一次，编码全部）
    print("[precompute] 加载 text encoder + embeddings processor（12B gemma，只加载一次）…", flush=True)
    te = load_text_encoder(gemma_model_path=E["CW_GEMMA"], device=device, dtype=torch.bfloat16, load_in_8bit=False)
    ep = load_embeddings_processor(
        checkpoint_path=embedding_weight_paths(E["CW_CKPT"], E["CW_GEMMA"]),
        gemma_model_path=E["CW_GEMMA"], device=device, dtype=torch.bfloat16)

    @torch.no_grad()
    def enc(prompt_text):
        """一个 prompt → {video, audio}（cpu）。与 runner/make_prompt_encoder 完全一致。"""
        hs, mask = te.encode([prompt_text])[0]
        out = ep.process_hidden_states(hs, mask)
        return {"video": out.video_encoding.cpu(),
                "audio": (out.audio_encoding.cpu() if out.audio_encoding is not None else None)}

    # 共享 negative（空串，与 deploy/make_prompt_encoder 一致）
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
            print(f"  [{i}/{len(todo)}] {r['clip_id']}  ({(time.time()-t0)/i:.1f}s/条)", flush=True)

    print(f"\n[precompute] 完成 {ok} 条 → {out_dir}")
    print("  推理端用 --prompt-embeds-dir 指到这里即可复用（deploy_pipeline / latent_rollout 支持后）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
