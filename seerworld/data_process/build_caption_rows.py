"""把 captions.jsonl 摊平成"一行一段文本"的清单，供 process_captions.py 编码。

process_captions.py 只用 media_path 做【输出 .pt 的命名】—— `_load_caption_data_from_jsonl`
只做路径字符串操作，不检查文件是否存在（已核实）。所以这里伪造 `<set>/<num>__seg<i>.mp4`，
让每一段拿到自己的 embedding 文件。

★ 它内部是 `caption_data[output_path] = caption`，**按输出路径做 key 的字典**，
重复路径会静默覆盖而不报错。所以命名必须唯一 —— 本脚本最后会显式校验这一点。

用法（在 captions.jsonl 全部打完之后跑）：
    python build_caption_rows.py --captions work/captions.jsonl --out work/caption_rows.jsonl
"""
import argparse, json, collections
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    # global_prompt 一起编：它只占总量的 1/4.5，而整片条件化的对照实验要用到它，
    # 现在顺手编掉比之后为了一个对照再起一次 GPU 任务便宜。
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
        raise SystemExit(f"!! {len(dup)} 个重复 media_path，会被静默覆盖。例: {dup[:3]}")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{n_clip} 条 clip，段数分布 {dict(per_nseg)} → {len(rows)} 行文本")
    print(f"  写入 {out}")
    print(f"  重复 media_path: 0（已校验）")


if __name__ == "__main__":
    main()
