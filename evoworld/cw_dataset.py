"""Windowed latent dataset for EvoWorld training."""
from __future__ import annotations
import json, math
from pathlib import Path


TEMPORAL_FACTOR = 8
FPS = 24.0

PREFIX_LAT = 4
TARGET_LAT = 11
WINDOW_LAT = PREFIX_LAT + TARGET_LAT
REF_LAT = 2


def latent_frames(k: int) -> tuple[int, int]:
    if k == 0:
        return 0, 0
    return TEMPORAL_FACTOR * (k - 1) + 1, TEMPORAL_FACTOR * k


def latents_fully_before(t_s: float) -> int:
    f = int(math.floor(t_s * FPS))
    n = 0
    while n < 4096:
        a, b = latent_frames(n)
        if b > f:
            break
        n += 1
    return n


class OoSWindowIndex:

    def __init__(self, work: str | Path, split: str | None = "train",
                 prefix_lat: int = PREFIX_LAT, target_lat: int = TARGET_LAT,
                 ref_lat: int = REF_LAT, require_subject_on_revisit: bool = True,
                 ref_mode: str = "head"):
        work = Path(work)
        self.prefix_lat, self.target_lat, self.ref_lat = prefix_lat, target_lat, ref_lat
        self.window_lat = prefix_lat + target_lat
        self.require_subject_on_revisit = require_subject_on_revisit


        assert ref_mode in ("head", "exit"), ref_mode
        self.ref_mode = ref_mode
        self.n_seg_s_fallback = 0

        caps = {r["clip_id"]: r for r in self._jl(work / "captions.jsonl")}


        man = {r["clip_id"]: r for r in self._jl(work / "manifest.jsonl")
               if r.get("keep", True) is not False}


        ph = ({r["clip_id"]: r for r in self._jl(work / "camera_phases.jsonl")}
              if (work / "camera_phases.jsonl").is_file() else {})
        tim = {r["clip_id"]: r for r in self._jl(work / "subject_timing.jsonl")} \
            if (work / "subject_timing.jsonl").is_file() else {}


        if ref_mode == "exit" and not tim and not any(
                isinstance((r.get("subject_timing") or {}).get("fully_out_s"), (int, float))
                for r in man.values()):
            raise FileNotFoundError(
                "ref_mode='exit' requires subject_timing.fully_out_s in the manifest "
                f"or {work}/subject_timing.jsonl. Use ref_mode='head' otherwise.")

        self.samples: list[dict] = []
        self.dropped: dict[str, int] = {}
        for cid, m in man.items():
            if split and m.get("split") != split:
                continue
            self._add_clip(cid, m, caps.get(cid), ph.get(cid), tim.get(cid))

    @staticmethod
    def _jl(p: Path):
        with open(p, encoding="utf-8") as f:
            for l in f:
                try:
                    yield json.loads(l)
                except Exception:
                    continue

    def _drop(self, why: str, n: int = 1):
        self.dropped[why] = self.dropped.get(why, 0) + n

    def _add_clip(self, cid, m, cap, phase, tim):
        if cap is None or not cap.get("segment_prompts"):
            return self._drop("no_caption")


        emb = m.get("phase") or {}
        if emb.get("arrival_s") is not None:
            phase = emb
        if phase is None:
            return self._drop("no_camera_phase")


        nseg = cap.get("nseg") or len(cap["segment_prompts"])
        seg_s = cap.get("seg_s")
        if seg_s is None:


            seg_s = 3.375 if m.get("duration_s") == 10 else 3.75
            self.n_seg_s_fallback += 1
        if nseg != len(cap["segment_prompts"]):
            return self._drop("nseg_mismatch")

        n_lat = self._n_lat(m)
        if n_lat is None:
            return self._drop("no_frame_count")
        if n_lat < self.window_lat:
            return self._drop("clip_too_short")


        if self.ref_mode == "head":
            ref_a, ref_b = 0, min(self.ref_lat, n_lat)
        else:


            out_s = (m.get("subject_timing") or {}).get("fully_out_s")
            if not isinstance(out_s, (int, float)):
                out_s = ((tim or {}).get("gemini") or {}).get("subject_fully_out_s")
            if not isinstance(out_s, (int, float)):
                return self._drop("no_fully_out")
            ref_b = min(latents_fully_before(float(out_s)), n_lat)
            ref_a = max(0, ref_b - self.ref_lat)
        if ref_b - ref_a < 1:
            return self._drop("no_reference")


        arrival = phase.get("arrival_s")
        rev_seg = None
        if isinstance(arrival, (int, float)) and seg_s > 0:
            rev_seg = min(int(float(arrival) // seg_s), nseg - 1)
        on_screen = cap.get("subject_on_screen") or []

        n_added = 0
        for s in range(0, n_lat - self.window_lat + 1):
            t_a = s + self.prefix_lat
            t_b = s + self.window_lat


            if t_a < ref_b:
                self._drop("ref_overlaps_target")
                continue
            seg = self._seg_of(t_a, t_b, nseg, seg_s)

            if (self.require_subject_on_revisit and rev_seg is not None and seg == rev_seg
                    and seg < len(on_screen) and on_screen[seg] is not True):
                self._drop("subject_absent_on_revisit")
                continue
            self.samples.append(dict(clip_id=cid, win=s, seg=seg,
                                     ref=(ref_a, ref_b), n_lat=n_lat,
                                     is_revisit=(seg == rev_seg)))
            n_added += 1
        if n_added == 0:
            self._drop("clip_yielded_nothing")

    @staticmethod
    def _n_lat(m):
        f = m.get("n_frames") or m.get("frames")
        if isinstance(f, int) and f >= 1:
            return 1 + (f - 1) // TEMPORAL_FACTOR
        d = m.get("duration_s")

        if d == 10:
            return 31
        if d == 15:
            return 46
        return None

    def _seg_of(self, t_a: int, t_b: int, nseg: int, seg_s: float) -> int:
        f0 = latent_frames(t_a)[0]
        f1 = latent_frames(t_b - 1)[1]
        best, best_ov = 0, -1
        for i in range(nseg):
            a = i * seg_s * FPS
            b = (i + 1) * seg_s * FPS
            ov = min(f1 + 1, b) - max(f0, a)
            if ov > best_ov:
                best, best_ov = i, ov
        return best

    def __len__(self):
        return len(self.samples)


#   reference_latents/<set>/<num>.pt    reference(make_reference_latents.py)


try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except Exception:
    torch = None
    _TorchDataset = object


def _resolve_root(root: Path, probe_clip: str | None, pattern: str, label: str) -> Path:
    if probe_clip is None:
        return root
    for sub in ("", "videos"):
        cand = root / sub if sub else root
        if (cand / pattern.format(probe_clip)).is_file():
            return cand
    have = sorted(p.name for p in root.iterdir())[:8] if root.is_dir() else []
    raise FileNotFoundError(
        f"{label}: no matching tensors under {root} or {root / 'videos'}. "
        f"{pattern.format(probe_clip)}.\n"
        f"  Entries under {root}: {have}\n"
        f"  Point --{label.replace('_', '-')} to the directory containing <set>/<num>.pt files.")


class OoSWindowDataset(_TorchDataset):

    def __init__(self, work, latents_dir, conditions_dir, reference_dir="reference_latents",
                 scene_reference_dir=None,
                 split="train", cache_clips: int = 4, verbose: bool = True,
                 oversample_revisit: int = 1, prefix_noise: float = 0.0,
                 oversample_i2v_start: int = 1, i2v_start: bool = False, **index_kw):


        if torch is None:
            raise RuntimeError("PyTorch is required")
        self.prefix_noise = prefix_noise
        self.i2v_start = i2v_start
        self.index = OoSWindowIndex(work, split=split, **index_kw)


        if oversample_revisit > 1 and self.index.samples:
            rev = [s for s in self.index.samples if s.get("is_revisit")]
            self.index.samples = self.index.samples + rev * (oversample_revisit - 1)
            if verbose:
                print(f"[oos] revisit oversampling x{oversample_revisit}: +{len(rev)*(oversample_revisit-1):,} samples "
                      f"-> {len(self.index.samples):,} total (revisit ratio "
                      f"{sum(1 for s in self.index.samples if s.get('is_revisit'))/len(self.index.samples)*100:.0f}%)")


        if oversample_i2v_start > 1 and self.index.samples:
            fst = [s for s in self.index.samples if s.get("win") == 0]
            self.index.samples = self.index.samples + fst * (oversample_i2v_start - 1)
            if verbose:
                print(f"[oos] I2V-start oversampling x{oversample_i2v_start}: +{len(fst)*(oversample_i2v_start-1):,} samples "
                      f"-> {len(self.index.samples):,} total (first-window ratio "
                      f"{sum(1 for s in self.index.samples if s.get('win')==0)/len(self.index.samples)*100:.0f}%)")
        probe = self.index.samples[0]["clip_id"] if self.index.samples else None


        #                     -> latents/videos/<set>/<num>.pt


        #                     -> conditions/<set>/<num>__seg<i>.pt


        self.latents = _resolve_root(Path(latents_dir), probe, "{}.pt", "latents")
        self.reference = Path(reference_dir)
        self.ref_root = _resolve_root(self.reference, probe, "{}.pt", "reference_latents")
        self.conditions = _resolve_root(Path(conditions_dir), probe, "{}__seg0.pt", "conditions")


        self.ref_key = self.reference.name

        self.scene_reference = None
        self.scene_ref_root = None
        self.scene_ref_key = None
        if scene_reference_dir:
            self.scene_reference = Path(scene_reference_dir)
            self.scene_ref_root = _resolve_root(self.scene_reference, probe, "{}.pt",
                                                "scene_reference_latents")
            self.scene_ref_key = self.scene_reference.name
            if self.scene_ref_key == self.ref_key:
                raise RuntimeError(
                    f"scene_reference_dir and reference_dir share the key {self.scene_ref_key!r}; "
                    "use distinct directory names.")
        self._cache: dict = {}
        self._cache_max = max(1, cache_clips)
        if verbose:
            for nm, p in (("latents", self.latents), ("conditions", self.conditions),
                          ("reference", self.ref_root)):
                print(f"[oos] {nm:11s} -> {p}")
            if self.scene_ref_root is not None:
                print(f"[oos] {'scene_ref':11s} -> {self.scene_ref_root}  (key='{self.scene_ref_key}')")

    def __len__(self):
        return len(self.index.samples)

    def _load_clip(self, clip_id):


        if clip_id in self._cache:
            return self._cache[clip_id]
        d = torch.load(self.latents / f"{clip_id}.pt", map_location="cpu", weights_only=True)
        if len(self._cache) >= self._cache_max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[clip_id] = d
        return d

    def __getitem__(self, i):
        s = self.index.samples[i]
        cid, w = s["clip_id"], s["win"]
        clip = self._load_clip(cid)
        lat = clip["latents"]                      # [C, F, H, W]
        win = lat[:, w : w + self.index.window_lat].contiguous()
        if win.shape[1] != self.index.window_lat:
            raise RuntimeError(f"{cid} window {w} contains {win.shape[1]} latents; "
                               f"index n_lat={s['n_lat']} but file has {lat.shape[1]}")


        if self.prefix_noise > 0:
            p = float(self.prefix_noise)
            pf = win[:, :PREFIX_LAT]
            win = win.clone()
            win[:, :PREFIX_LAT] = (1.0 - p) * pf + p * torch.randn_like(pf)

        txt = torch.load(self.conditions / f"{cid}__seg{s['seg']}.pt",
                         map_location="cpu", weights_only=True)
        ref = torch.load(self.ref_root / f"{cid}.pt", map_location="cpu", weights_only=True)


        if self.i2v_start:
            ref = dict(ref)
            ref["latents"] = ref["latents"][:, :1]
            ref["num_frames"] = 1

        return {
            "video_latents": {
                "latents": win,

                "num_frames": self.index.window_lat,
                "height": int(clip["height"]),
                "width": int(clip["width"]),
                "fps": clip["fps"],
            },
            "conditions": txt,
            self.ref_key: ref,
            **({self.scene_ref_key: torch.load(self.scene_ref_root / f"{cid}.pt",
                                               map_location="cpu", weights_only=True)}
               if self.scene_ref_root is not None else {}),

            "meta": {"clip_id": cid, "win": w, "seg": s["seg"], "is_revisit": s["is_revisit"],
                     "is_first": (w == 0)},


            "is_revisit": int(bool(s["is_revisit"])),
            "idx": i,
        }


def collate(batch):
    if torch is None:
        raise RuntimeError("PyTorch is required")

    def pad_text(items):
        keys = [k for k in ("video_prompt_embeds", "audio_prompt_embeds") if k in items[0]]
        L = max(it[keys[0]].shape[0] for it in items)
        out = {}
        for k in keys:
            out[k] = torch.stack([
                torch.cat([it[k], it[k].new_zeros(L - it[k].shape[0], *it[k].shape[1:])])
                if it[k].shape[0] < L else it[k] for it in items])

        if "prompt_attention_mask" in items[0]:
            m = [it["prompt_attention_mask"] for it in items]
            out["prompt_attention_mask"] = torch.stack([
                torch.cat([x, x.new_zeros(L - x.shape[0])]) if x.shape[0] < L else x for x in m])
        return out

    def stack_latents(items):
        return {
            "latents": torch.stack([it["latents"] for it in items]),


            "num_frames": torch.tensor([it["num_frames"] for it in items]),
            "height": torch.tensor([it["height"] for it in items]),
            "width": torch.tensor([it["width"] for it in items]),
            "fps": torch.tensor([float(it["fps"]) for it in items]),
        }


    ref_keys = [k for k in batch[0] if k not in
                ("video_latents", "conditions", "meta", "idx", "is_revisit")]
    out = {
        "video_latents": stack_latents([b["video_latents"] for b in batch]),
        "conditions": pad_text([b["conditions"] for b in batch]),
        "meta": [b["meta"] for b in batch],
        "idx": torch.tensor([b["idx"] for b in batch]),

        "is_revisit": torch.tensor([int(b.get("is_revisit", 0)) for b in batch]),
    }
    for k in ref_keys:
        out[k] = stack_latents([b[k] for b in batch])
    return out


def _selftest(work="work", split=None, ref_mode="head"):
    import collections
    idx = OoSWindowIndex(work, split=split, ref_mode=ref_mode)
    print(f"split={split} ref_mode={idx.ref_mode} samples={len(idx):,} "
          f"seg_s_fallback={idx.n_seg_s_fallback:,}")
    print("  Drop reasons:")
    for k, v in sorted(idx.dropped.items(), key=lambda kv: -kv[1]):
        print(f"    {k:28s} {v:,}")
    per = collections.Counter(s["clip_id"] for s in idx.samples)
    seg = collections.Counter((s["seg"], s["is_revisit"]) for s in idx.samples)
    print(f"  Clips={len(per):,}; windows per clip min/median/max = "
          f"{min(per.values())}/{sorted(per.values())[len(per)//2]}/{max(per.values())}")
    nrev = sum(v for (_, r), v in seg.items() if r)
    print(f"  Revisit windows={nrev:,} ({nrev/max(1,len(idx))*100:.1f}%)")
    print("  Segment distribution:", {k: v for k, v in sorted(seg.items())})
    print("\n  First three samples:")
    for s in idx.samples[:3]:
        t_a = s["win"] + PREFIX_LAT
        f0, f1 = latent_frames(t_a)[0], latent_frames(s["win"] + WINDOW_LAT - 1)[1]
        print(f"    {s['clip_id']}  ref={s['ref']}  prefix=[{s['win']},{t_a})  "
              f"target=[{t_a},{s['win']+WINDOW_LAT}) source_frames={f0}-{f1} "
              f"({f0/FPS:.2f}-{f1/FPS:.2f}s)  seg={s['seg']} revisit={s['is_revisit']}")


if __name__ == "__main__":


    import argparse
    ap = argparse.ArgumentParser(description="Validate dataset metadata without loading latents")
    ap.add_argument("--work", required=True,
                    help="Directory containing captions.jsonl and manifest.jsonl")
    ap.add_argument("--split", default=None, help="Optional dataset split")
    ap.add_argument("--ref-mode", default="head", choices=["head", "exit"])
    a = ap.parse_args()
    w = Path(a.work)


    need = ["captions.jsonl", "manifest.jsonl"]


    missing = [f for f in need if not (w / f).is_file()]
    if missing:
        raise SystemExit(f"Missing under {w}: {', '.join(missing)}")
    _selftest(work=a.work, split=a.split, ref_mode=a.ref_mode)
