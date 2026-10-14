"""从整片 latent 切出 reference，写成 trainer 要的独立文件。

    python make_reference_latents.py --latents ./latents --out ./reference_latents

【为什么要单独落盘】ReferenceConditionConfig 是按 `latents_dir` 读的，而且
`_apply_reference_condition` 里取的是 `batch[config.latents_dir]` —— 这个字符串既是目录名
也是 batch 的 key。它不是运行时从整片 latent 里切的，所以必须预先存成一份。
**但不需要重新编码**：直接从已有的整片 latent 上切前 N 个 latent 即可。

【为什么取片头而不是"主体出画前"】两个理由，第二个是决定性的：
  1. 实测：出画前取会让 ref 边界推到 latent 4-18，而 target 起点是 win+4，
     于是靠前的窗口全因 ref/target 重叠被丢弃 —— seg0 样本从 34,612 掉到 9,641。
  2. ★ `_apply_reference_condition` 给 ref 生成位置编码时用的是
     `_get_video_positions(num_frames=cond.num_frames, ...)`，**从 frame 0 开始计**。
     片头取的 ref 真实位置就是 frame 0，位置编码是**字面正确**的；出画前取则等于告诉
     模型"这段在 t=0"而它实际在 t=1.5-3.0s，是一句谎话。

输出 schema 与 process_videos.py 一致（`{"latents","num_frames","height","width","fps"}`，
其中 num_frames 是 **latent** 帧数），否则 `_patchify_latent_data` 读不了。
"""
import argparse, os, sys
from pathlib import Path

import torch

REF_LAT = 2


def main():
    ap = argparse.ArgumentParser()
    # 默认值来自 .env 导出的环境变量。
    ap.add_argument("--latents", default=os.environ.get("CW_LATENTS"),
                    required="CW_LATENTS" not in os.environ,
                    help="整片 latent 目录（含 <set>/<num>.pt）")
    ap.add_argument("--out", default=os.environ.get("CW_REFERENCE"),
                    required="CW_REFERENCE" not in os.environ,
                    help="输出目录，必须与 yaml 里 reference 条件的 latents_dir 同名")
    ap.add_argument("--ref-lat", type=int, default=REF_LAT)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    src, dst = Path(a.latents), Path(a.out)
    files = sorted(src.rglob("*.pt"))
    if not files:
        sys.exit(f"!! {src} 下没有 .pt")
    print(f"{len(files)} 个整片 latent → ref {a.ref_lat} latents")

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
            print(f"  !! {rel} latents 维度是 {tuple(lat.shape)}，期望 [C,F,H,W]")
            bad += 1
            continue
        if lat.shape[1] < a.ref_lat:
            print(f"  !! {rel} 只有 {lat.shape[1]} 个 latent，不足 {a.ref_lat}")
            bad += 1
            continue
        ref = lat[:, : a.ref_lat].contiguous()
        rec = {
            "latents": ref,
            # num_frames 是 LATENT 帧数，不是像素帧数 —— process_videos.py 存的就是这个语义
            # （它把 video_latent_data["num_frames"] 直接写进去）。写成 8*N 会让
            # _get_video_positions 生成 8 倍长的位置序列，与 latents 的实际长度不符。
            "num_frames": int(a.ref_lat),
            "height": int(d["height"]),
            "width": int(d["width"]),
            # fps 必须带上：_apply_reference_condition 用 cond.fps 算位置。
            # 我们没做时间下采样，所以就是源 fps，reference_temporal_scale_factor 保持 1。
            "fps": d["fps"],
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".pt.tmp")
        torch.save(rec, tmp)
        tmp.replace(out)          # 原子写：中断不会留下半个文件被后续当成"已完成"跳过
        shapes[tuple(ref.shape)] = shapes.get(tuple(ref.shape), 0) + 1
        done += 1
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}  写 {done} 跳过 {skip} 异常 {bad}", flush=True)

    print(f"\n完成 写 {done} 跳过 {skip} 异常 {bad}")
    for s, n in sorted(shapes.items()):
        print(f"  shape {s}: {n}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
