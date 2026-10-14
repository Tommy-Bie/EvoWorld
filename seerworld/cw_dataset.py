"""OoS 滑窗数据集：在整片 latent 上切 (reference, prefix, target)，并挂上 target 所在相位的 caption。

为什么不用 trainer 自带的 PrecomputedDataset：它是**一文件一样本**的（`latents/x.pt` ↔
`conditions/x.pt`），而我们有 ~29 万个滑窗。落盘成 29 万个文件既浪费磁盘也浪费 IO，
所以在整片 latent 上切。输出字典结构与它保持一致，trainer 侧无需改动。

对应的 trainer 配置（`training_strategies/flexible.py`）：
    conditions = [PrefixConditionConfig(temporal_boundary=PREFIX_LAT),
                  ReferenceConditionConfig(latents_dir=...)]
prefix 段 timestep=0、不计 loss；reference 段拼在序列上、timestep=0、不计 loss、参与双向注意力。
"""
from __future__ import annotations
import json, math
from pathlib import Path

# ---- latent 网格 ----------------------------------------------------------------
# 实测自数据本身：10s = 241 帧 → 31 latents，15s = 361 帧 → 46 latents，
# 两者都满足 n_lat = 1 + (frames-1)/8。⇒ latent 0 只覆盖 frame 0，之后每个覆盖 8 帧。
# 这条不能写成 "1 latent = 8 帧"：那会让每个跨度都偏 1 帧，而偏移会一路传到相位判定里。
TEMPORAL_FACTOR = 8
FPS = 24.0

PREFIX_LAT = 4          # 与 §11 定型一致：4 lat = 1792 token
TARGET_LAT = 11         # 11 lat = 4928 token，合计 6720
WINDOW_LAT = PREFIX_LAT + TARGET_LAT
REF_LAT = 2             # reference 取几个 latent


def latent_frames(k: int) -> tuple[int, int]:
    """latent k 覆盖的源帧闭区间。"""
    if k == 0:
        return 0, 0
    return TEMPORAL_FACTOR * (k - 1) + 1, TEMPORAL_FACTOR * k


def latents_fully_before(t_s: float) -> int:
    """时刻 t_s 之前**完整结束**的 latent 个数（即可安全用作 reference 的上界）。

    取 "完整结束" 而不是 "包含 t_s"：reference 的语义是"主体还在画面里的样子"，
    跨过出画时刻的那个 latent 里主体已经半出画，混进去会让参考图本身是残缺的。
    """
    f = int(math.floor(t_s * FPS))
    n = 0
    while n < 4096:
        a, b = latent_frames(n)
        if b > f:
            break
        n += 1
    return n


class OoSWindowIndex:
    """纯元数据的索引构建 —— 不碰 latent 文件，所以可以在 Mac 上直接跑通验证。

    每个样本 = (clip_id, win_start_lat, seg_idx)：
      reference latents = [ref_a, ref_b)
      prefix   latents = [win_start, win_start+PREFIX_LAT)
      target   latents = [win_start+PREFIX_LAT, win_start+WINDOW_LAT)
      文本 = 该 clip 第 seg_idx 段的 caption
    """

    def __init__(self, work: str | Path, split: str | None = "train",
                 prefix_lat: int = PREFIX_LAT, target_lat: int = TARGET_LAT,
                 ref_lat: int = REF_LAT, require_subject_on_revisit: bool = True,
                 ref_mode: str = "head"):
        work = Path(work)
        self.prefix_lat, self.target_lat, self.ref_lat = prefix_lat, target_lat, ref_lat
        self.window_lat = prefix_lat + target_lat
        self.require_subject_on_revisit = require_subject_on_revisit
        # head : reference = latents [0, ref_lat)  —— 片头最干净的建立视角
        # exit : reference = 主体完全出画前最后 ref_lat 个 —— 离事件最近的"原状态"
        #
        # 默认 head，因为 exit 有一个实测出来的副作用：它把 ref 边界推到 latent 4-18，
        # 而 target 起点是 win+4，于是所有靠前的窗口都因 ref/target 重叠被丢掉 ——
        # 段分布实测变成 seg0 只剩 235 个而 seg2 有 12k，等于 seg0 几乎不被训练。
        # head 模式下 ref_b = ref_lat = 2，而 target 起点恒 >= prefix_lat = 4，
        # **永不重叠**，一个窗口都不丢。ref 与 prefix 会有重叠，但 prefix 本身也是干净的、
        # 不计 loss，所以那只是冗余而不是泄漏。
        assert ref_mode in ("head", "exit"), ref_mode
        self.ref_mode = ref_mode
        self.n_seg_s_fallback = 0

        caps = {r["clip_id"]: r for r in self._jl(work / "captions.jsonl")}
        # ★ keep 判据：显式 False 才算剔除，【字段缺失算保留】。
        # 两种 manifest 都要能读：全量版本显式包含 keep，而发布版本只包含
        # 筛选后条目且省略 keep。字段缺失因此也表示保留。
        man = {r["clip_id"]: r for r in self._jl(work / "manifest.jsonl")
               if r.get("keep", True) is not False}
        # 相位有两个可能来源，优先用 manifest 内嵌的那份：
        # 发布工具把 phase / subject_timing 直接写进 manifest（自包含设计），
        # 因此不依赖另外两个文件。camera_phases.jsonl 作为回退。
        # 光流相位的正主是 camera_phases.jsonl（step1 的直接产物）。
        # subject_timing.jsonl（step4）里也带一份 flow 的拷贝，但它是【可选】的 ——
        # 只有 ref_mode="exit" 需要它的 gemini.subject_fully_out_s。
        # 不能无条件要求 subject_timing.jsonl，否则自包含发布数据会被误判为
        # 缺少 timing，导致全部样本被静默过滤。
        ph = ({r["clip_id"]: r for r in self._jl(work / "camera_phases.jsonl")}
              if (work / "camera_phases.jsonl").is_file() else {})
        tim = {r["clip_id"]: r for r in self._jl(work / "subject_timing.jsonl")} \
            if (work / "subject_timing.jsonl").is_file() else {}
        # exit 模式需要"主体完全出画时刻"，它同样有两个来源：manifest 内嵌的
        # subject_timing.fully_out_s，或独立的 subject_timing.jsonl。两个都没有才是错。
        # 只查 tim 是否为空会把自包含 manifest 误判成缺文件。
        if ref_mode == "exit" and not tim and not any(
                isinstance((r.get("subject_timing") or {}).get("fully_out_s"), (int, float))
                for r in man.values()):
            raise FileNotFoundError(
                f"ref_mode='exit' 需要主体出画时刻，但 {work}/manifest.jsonl 里没有内嵌 "
                f"subject_timing.fully_out_s，{work}/subject_timing.jsonl 也不存在。"
                f"ref_mode='head'（默认）不需要它。")

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
        # manifest 内嵌的 phase 优先。判据用 arrival_s 是否存在，而不是 phase 键是否存在 ——
        # step5_shard 用 c.get(k) 填这个字典，cam 里没有该 clip 时会写成一堆 None，
        # 那种"存在但全空"的字典必须视为无效，否则 revisit 段判定会静默退化成第 0 段。
        emb = m.get("phase") or {}
        if emb.get("arrival_s") is not None:
            phase = emb
        if phase is None:
            return self._drop("no_camera_phase")

        # nseg / seg_s 必须从记录里读。文件里同时存在 10s 的 3 段和 15s 的 4 段两种布局，
        # 任何硬编码都会把 15s 的 caption 错位挂到滑窗上 —— 而错位不会报错，
        # 它只会安静地训出一个把相位学反的模型（§12.3 已经踩过一次这个坑）。
        nseg = cap.get("nseg") or len(cap["segment_prompts"])
        seg_s = cap.get("seg_s")
        if seg_s is None:
            # 10s 那批是在 nseg/seg_s 字段加进来之前打的。规则是确定的（seg_layout），
            # 所以按时长回填，而不是把 6413 条好数据丢掉。写在这里而不是改
            # captions.jsonl，是因为 15s 的重打任务此刻正在往那个文件追加，改它会撞车。
            seg_s = 3.375 if m.get("duration_s") == 10 else 3.75
            self.n_seg_s_fallback += 1
        if nseg != len(cap["segment_prompts"]):
            return self._drop("nseg_mismatch")

        n_lat = self._n_lat(m)
        if n_lat is None:
            return self._drop("no_frame_count")
        if n_lat < self.window_lat:
            return self._drop("clip_too_short")

        # reference：主体完全出画之前、完整结束的 latent 里取最后 ref_lat 个。
        # 用 gemini.subject_fully_out_s 而不是 flow.pan_away_start —— 后者中位只有
        # 0.25s/0.58s，之前只有 1 个完整 latent，参考几乎等于单帧；而摇镜头刚起步时
        # 主体仍在画面内，用出画时刻能拿到 5(10s)/9(15s) 个 latent。
        if self.ref_mode == "head":
            ref_a, ref_b = 0, min(self.ref_lat, n_lat)
        else:
            # 字段名两处不同：manifest 内嵌的是 subject_timing.fully_out_s，
            # 独立文件里的是 gemini.subject_fully_out_s。两个都要认。
            out_s = (m.get("subject_timing") or {}).get("fully_out_s")
            if not isinstance(out_s, (int, float)):
                out_s = ((tim or {}).get("gemini") or {}).get("subject_fully_out_s")
            if not isinstance(out_s, (int, float)):
                return self._drop("no_fully_out")
            ref_b = min(latents_fully_before(float(out_s)), n_lat)
            ref_a = max(0, ref_b - self.ref_lat)
        if ref_b - ref_a < 1:
            return self._drop("no_reference")

        # revisit 段 = 含 arrival 的那段（相机真正回到主体的时刻，取自光流）。
        # arrival 直接从 camera_phases.jsonl 取（顶层字段），不再绕 subject_timing 的 flow 拷贝。
        arrival = phase.get("arrival_s")
        rev_seg = None
        if isinstance(arrival, (int, float)) and seg_s > 0:
            rev_seg = min(int(float(arrival) // seg_s), nseg - 1)
        on_screen = cap.get("subject_on_screen") or []

        n_added = 0
        for s in range(0, n_lat - self.window_lat + 1):
            t_a = s + self.prefix_lat
            t_b = s + self.window_lat
            # ★ reference 不能与 target 重叠。reference token 是干净的、timestep=0、不计 loss，
            # 一旦落进 target 区间就等于把答案直接摆在输入里，loss 会好看而模型什么都没学到。
            if t_a < ref_b:
                self._drop("ref_overlaps_target")
                continue
            seg = self._seg_of(t_a, t_b, nseg, seg_s)
            # 末段（回望）里主体必须真的在画面里，否则这条窗口教的是"摇回去什么也没发生"。
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
        # 回退：10s=241 帧→31、15s=361 帧→46，两者都实测确认过。
        if d == 10:
            return 31
        if d == 15:
            return 46
        return None

    def _seg_of(self, t_a: int, t_b: int, nseg: int, seg_s: float) -> int:
        """target 落在哪一段：按**源帧重叠最大**判定，不用中点。

        target 跨 3.67s 而段长 3.375/3.75s，所以它经常横跨两段；中点法会在边界上
        突然整段跳变，重叠最大法给出的是"这段文字描述的动作占 target 多少"。
        """
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


# ---- 张量装配 ------------------------------------------------------------------
# 依赖三份预处理产物：
#   latents/<set>/<num>.pt              整片 latent（process_videos.py）
#   conditions/<set>/<num>__seg<i>.pt   每段文本 embedding（encode_text_launch.sh）
#   reference_latents/<set>/<num>.pt    reference（make_reference_latents.py）
#
# torch 放在 try 里：元数据自检（_selftest / run.sh index）不需要 torch，
# 在没装 torch 的机器上也该能跑。
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except Exception:
    torch = None
    _TorchDataset = object


def _resolve_root(root: Path, probe_clip: str | None, pattern: str, label: str) -> Path:
    """找出 root 下真正放着 <clip_id> 文件的那一层。

    候选只有两个：root 本身，或 root/videos。不做通用递归搜索 ——
    在 12118 个文件的目录树上乱猜会既慢又可能命中同名的无关文件。
    探测不到就抛错并列出 root 下实际有什么，让人一眼看出层级差在哪。
    """
    if probe_clip is None:
        return root
    for sub in ("", "videos"):
        cand = root / sub if sub else root
        if (cand / pattern.format(probe_clip)).is_file():
            return cand
    have = sorted(p.name for p in root.iterdir())[:8] if root.is_dir() else []
    raise FileNotFoundError(
        f"{label}: 在 {root} 和 {root / 'videos'} 下都找不到 "
        f"{pattern.format(probe_clip)}。\n"
        f"  {root} 下现有: {have}\n"
        f"  用 --{label.replace('_', '-')} 指向真正含 <set>/<num>.pt 的那一层。")


class OoSWindowDataset(_TorchDataset):
    """输出的 batch 键名必须与 flexible strategy 读的对上，trainer 侧零改动。

    ★ 键名的真源是 FlexibleStrategyConfig.get_data_sources()（flexible.py:187），不是
    PrecomputedDataset._normalize_data_sources()。后者的默认映射是
        {"latents": "latent_conditions", "conditions": "text_conditions"}
    而 flexible 用的是
        {"conditions": "conditions", <video.latents_dir>: "video_latents",
         <reference.latents_dir>: <同名>}
    我第一版照 PrecomputedDataset 写成了 latent_conditions / text_conditions，
    结果是 trainer.py:368 的 `batch["conditions"]` 抛 KeyError —— 而且要等
    dataloader 真吐出第一个 batch 才炸。

      conditions        ← 文本 embedding（trainer.py:368、flexible.py:370）
      video_latents     ← 窗口 latent（flexible.py:362 读 batch[f"{modality}_latents"]）
      reference_latents ← reference（flexible.py:618 读 batch[config.latents_dir]，
                          那个字符串既是目录名也是 batch 的 key）

    ★ num_frames / height / width / fps 必须是【能被 [0].item() 取的】——
    _patchify_latent_data(flexible.py:580) 就是这么读的。batch_size=1 时默认 collate
    会把 python int 变成 tensor([v])，所以这里直接给 int 是对的；
    自己写 collate 时必须记得转成张量（见下面的 collate）。
    """

    def __init__(self, work, latents_dir, conditions_dir, reference_dir="reference_latents",
                 scene_reference_dir=None,
                 split="train", cache_clips: int = 4, verbose: bool = True,
                 oversample_revisit: int = 1, prefix_noise: float = 0.0,
                 oversample_i2v_start: int = 1, i2v_start: bool = False, **index_kw):
        # scene_reference_dir: 可选的第二 reference —— pan peak 那 1 个 latent。
        # 由 make_scene_reference_latents.py 从整片 latent 里按 refs.jsonl 切出来。
        # 目录名（basename）必须与 yaml 里对应的 reference 条件 latents_dir 一致，
        # 因为它同时是 batch 的 key（同 ref_key 的逻辑）。传 None = 不启用（老行为）。
        if torch is None:
            raise RuntimeError("需要 torch")
        self.prefix_noise = prefix_noise      # 0 = 完全不动（与原行为一致）
        self.i2v_start = i2v_start            # 方案 A2：reference 全程用首帧（1 latent）
        self.index = OoSWindowIndex(work, split=split, **index_kw)
        # revisit 过采样：离视演化是核心信号，但 revisit 窗口只占 ~22.8%。把 is_revisit 的样本
        # 重复 (N-1) 次，让每步更可能是"回望发现变化"。只是重复样本，不改模型/训练逻辑，
        # 训崩风险极低。N=1 时完全不动（默认）。
        if oversample_revisit > 1 and self.index.samples:
            rev = [s for s in self.index.samples if s.get("is_revisit")]
            self.index.samples = self.index.samples + rev * (oversample_revisit - 1)
            if verbose:
                print(f"[oos] revisit 过采样 ×{oversample_revisit}：+{len(rev)*(oversample_revisit-1):,} 个样本 "
                      f"→ 共 {len(self.index.samples):,}（revisit 占比 "
                      f"{sum(1 for s in self.index.samples if s.get('is_revisit'))/len(self.index.samples)*100:.0f}%）")
        # I2V 起跳过采样：第一段窗口（w=0）是"仅 image condition"，只占每 clip 一个，会被后续
        # 4-latent-prefix 窗口淹没（~2-3%），需过采样补到合理占比。和 revisit 一样只重复样本。
        # oversample_i2v_start=1（默认）时完全不动，与原行为一致。
        if oversample_i2v_start > 1 and self.index.samples:
            fst = [s for s in self.index.samples if s.get("win") == 0]
            self.index.samples = self.index.samples + fst * (oversample_i2v_start - 1)
            if verbose:
                print(f"[oos] I2V 起跳过采样 ×{oversample_i2v_start}：+{len(fst)*(oversample_i2v_start-1):,} 个样本 "
                      f"→ 共 {len(self.index.samples):,}（首帧窗口占比 "
                      f"{sum(1 for s in self.index.samples if s.get('win')==0)/len(self.index.samples)*100:.0f}%）")
        probe = self.index.samples[0]["clip_id"] if self.index.samples else None
        # ★ 三个目录的层级不一致，必须探测而不是假定：
        #   latents/          由 process_videos.py 产出，它【镜像 dataset_videos.jsonl 里的
        #                     相对路径】，而那里写的是 videos/<set>/<num>.mp4
        #                     ⇒ latents/videos/<set>/<num>.pt
        #   reference_latents/ 由 make_reference_latents.py 从 latents/ 用 rglob 相对路径复制
        #                     ⇒ 同样带 videos/ 这一层
        #   conditions/       由 process_captions.py 产出，media_path 是
        #                     <set>/<num>__seg<i>.mp4（build_caption_rows.py 写的，无 videos/）
        #                     ⇒ conditions/<set>/<num>__seg<i>.pt
        # check_meta 用 rglob 数个数，所以三个目录的计数都对得上，它抓不到这个错 ——
        # 只有真去取某一条的文件才会暴露。
        self.latents = _resolve_root(Path(latents_dir), probe, "{}.pt", "latents")
        self.reference = Path(reference_dir)
        self.ref_root = _resolve_root(self.reference, probe, "{}.pt", "reference_latents")
        self.conditions = _resolve_root(Path(conditions_dir), probe, "{}__seg0.pt", "conditions")
        # ref_key 用【用户给的目录名】而不是探测后的路径名：它必须等于 yaml 里的
        # latents_dir（_apply_reference_condition 取 batch[config.latents_dir]），
        # 而探测可能往下钻了一层，那层的名字是 videos，不是 reference_latents。
        self.ref_key = self.reference.name
        # 第二个 reference（scene anchor，可选）。同样探测目录并保留 basename 作 batch key。
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
                    f"scene_reference_dir 的目录名 '{self.scene_ref_key}' 和 reference_dir 相同，"
                    f"两个 reference 会覆盖同一个 batch key；给 scene reference 另起个目录名。")
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
        # 连续窗口共享同一个整片文件。随机 shuffle 时命中率低，但顺序遍历（val、
        # 或按 clip 分组的 sampler）时能省掉绝大部分 IO。缓存故意做得很小：
        # 整片 latent 是 4-6MB，缓存大了就是在和 dataloader worker 抢内存。
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
            raise RuntimeError(f"{cid} 窗口 {w} 只切到 {win.shape[1]} 个 latent，"
                               f"索引里记的 n_lat={s['n_lat']} 与文件实际 {lat.shape[1]} 不符")

        # ★ prefix noise（开关，默认 0=不动，行为和原来完全一致）：给窗口的前 PREFIX_LAT 个
        # latent 按 flow-matching 插值到噪声 level p —— prefix' = (1-p)*clean + p*noise。
        # 训练时 flexible 仍把这几个当 clean prefix（timestep=0），但内容已带噪，
        # 于是模型学会"名义 clean 但实际不完美的 prefix"也能续对 —— 正是推理自回归时
        # prefix 是【生成的、不完美】的那个 gap（exposure bias）。缓解 rollout 后段漂移。
        # 在 dataset 层做，不碰上游 flexible.py。
        if self.prefix_noise > 0:
            p = float(self.prefix_noise)
            pf = win[:, :PREFIX_LAT]
            win = win.clone()
            win[:, :PREFIX_LAT] = (1.0 - p) * pf + p * torch.randn_like(pf)

        txt = torch.load(self.conditions / f"{cid}__seg{s['seg']}.pt",
                         map_location="cpu", weights_only=True)
        ref = torch.load(self.ref_root / f"{cid}.pt", map_location="cpu", weights_only=True)
        # ★ 方案 A2（I2V-START）：reference 全程用首帧（= 整片的 latent[0]），不再是前 ref_lat 个。
        # 用户部署时只给首帧图，所以身份记忆也只能来自那一帧。只在开关开时裁，默认不动。
        # 裁的是 latents 与 num_frames（_apply_reference_condition 用 num_frames 生成位置编码，
        # 必须与真实 latent 数一致，否则位置长度对不上）。
        if self.i2v_start:
            ref = dict(ref)                       # 不改缓存里的原对象
            ref["latents"] = ref["latents"][:, :1]
            ref["num_frames"] = 1                 # 1 latent = 首帧

        return {
            "video_latents": {
                "latents": win,
                # num_frames 是 LATENT 帧数（process_videos.py 的语义），不是像素帧数
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
            # 训练时不用，但可视化和评测要靠它定位样本
            "meta": {"clip_id": cid, "win": w, "seg": s["seg"], "is_revisit": s["is_revisit"],
                     "is_first": (w == 0)},
            # ★ IAA 训练要读这个 —— top-level bool tensor（0/1），flexible.py 里检查
            # batch.get("is_revisit") 决定这条样本是否加 attention log-bias。放 top-level
            # 而不是嵌在 meta 里，是因为 default_collate 会把 int 转成 tensor([v])，语义好用；
            # meta 是 list-of-dict、default_collate 不会 stack。
            "is_revisit": int(bool(s["is_revisit"])),
            "idx": i,
        }


def collate(batch):
    """★ 文本 embedding 是变长的，必须自己 pad —— 默认 collate 会直接抛 shape 不匹配。

    process_captions.py 里没有任何 `max_length` / `padding` 参数，它逐条编码逐条存盘
    （Gemma tokenizer 不支持批量分词，batch_size>1 会被静默改成 1），所以每条 caption 的
    `video_prompt_embeds` 长度就是它自己的 token 数。

    ⚠️ 当前训练用 batch_size=1，走的是 DataLoader 的默认 collate，【本函数没有被接上】——
    trainer 建 DataLoader 时不接受 collate_fn。要开 batch>1 必须同时改 trainer 那一处。

    latent 侧不需要 pad：所有窗口都是 window_lat 个 latent、同一分辨率，形状天然一致。
    """
    if torch is None:
        raise RuntimeError("需要 torch")

    def pad_text(items):
        keys = [k for k in ("video_prompt_embeds", "audio_prompt_embeds") if k in items[0]]
        L = max(it[keys[0]].shape[0] for it in items)
        out = {}
        for k in keys:
            out[k] = torch.stack([
                torch.cat([it[k], it[k].new_zeros(L - it[k].shape[0], *it[k].shape[1:])])
                if it[k].shape[0] < L else it[k] for it in items])
        # mask 用 0 补：pad 出来的位置必须是"无效"，补 1 会让模型去 attend 全零向量。
        if "prompt_attention_mask" in items[0]:
            m = [it["prompt_attention_mask"] for it in items]
            out["prompt_attention_mask"] = torch.stack([
                torch.cat([x, x.new_zeros(L - x.shape[0])]) if x.shape[0] < L else x for x in m])
        return out

    def stack_latents(items):
        return {
            "latents": torch.stack([it["latents"] for it in items]),
            # ★ 这四个必须是【张量】而不是 python 标量：_patchify_latent_data
            # （flexible.py:580）是按 latent_data["num_frames"][0].item() 取的。
            # 给 int 会在 [0] 处抛 TypeError。默认 collate 恰好会做这个转换，
            # 所以 __getitem__ 里给 int 是对的，而自己写 collate 就必须显式转。
            #
            # 整批的这几个值必须一致，否则 patchify 出来的序列长度对不上 ——
            # 这也是 trainer 那条"多分辨率 bucket 必须 batch size 1"警告的由来。
            "num_frames": torch.tensor([it["num_frames"] for it in items]),
            "height": torch.tensor([it["height"] for it in items]),
            "width": torch.tensor([it["width"] for it in items]),
            "fps": torch.tensor([float(it["fps"]) for it in items]),
        }

    # 所有非固定键都当作 reference（可能有多条 —— 主 ref + scene_ref 等）。
    # 每条按 latents_dir/basename 作 batch key，flexible 从 batch[config.latents_dir] 取。
    ref_keys = [k for k in batch[0] if k not in
                ("video_latents", "conditions", "meta", "idx", "is_revisit")]
    out = {
        "video_latents": stack_latents([b["video_latents"] for b in batch]),
        "conditions": pad_text([b["conditions"] for b in batch]),
        "meta": [b["meta"] for b in batch],
        "idx": torch.tensor([b["idx"] for b in batch]),
        # IAA: revisit 掩码，供 flexible.py 里 CW_USE_IAA=1 时决定加 log-bias 的样本
        "is_revisit": torch.tensor([int(b.get("is_revisit", 0)) for b in batch]),
    }
    for k in ref_keys:
        out[k] = stack_latents([b[k] for b in batch])
    return out


def _selftest(work="work", split=None, ref_mode="head"):
    import collections
    idx = OoSWindowIndex(work, split=split, ref_mode=ref_mode)
    print(f"split={split}  ref_mode={idx.ref_mode}  样本 {len(idx):,}"
          f"  (seg_s 回填 {idx.n_seg_s_fallback:,} 条)")
    print("  丢弃原因:")
    for k, v in sorted(idx.dropped.items(), key=lambda kv: -kv[1]):
        print(f"    {k:28s} {v:,}")
    per = collections.Counter(s["clip_id"] for s in idx.samples)
    seg = collections.Counter((s["seg"], s["is_revisit"]) for s in idx.samples)
    print(f"  覆盖 clip {len(per):,}，每条窗口数 min/中位/max = "
          f"{min(per.values())}/{sorted(per.values())[len(per)//2]}/{max(per.values())}")
    nrev = sum(v for (_, r), v in seg.items() if r)
    print(f"  revisit 窗口 {nrev:,} ({nrev/max(1,len(idx))*100:.1f}%)")
    print("  段分布:", {k: v for k, v in sorted(seg.items())})
    print("\n  头 3 个样本:")
    for s in idx.samples[:3]:
        t_a = s["win"] + PREFIX_LAT
        f0, f1 = latent_frames(t_a)[0], latent_frames(s["win"] + WINDOW_LAT - 1)[1]
        print(f"    {s['clip_id']}  ref={s['ref']}  prefix=[{s['win']},{t_a})  "
              f"target=[{t_a},{s['win']+WINDOW_LAT})  源帧 {f0}-{f1} "
              f"({f0/FPS:.2f}-{f1/FPS:.2f}s)  seg={s['seg']} revisit={s['is_revisit']}")


if __name__ == "__main__":
    # work 必须显式给：这个文件住在发布仓库里（SeerWorld/seerworld/），
    # 而三个 jsonl 位于单独的数据目录。默认成 "work" 只会抛一个 FileNotFoundError，
    # 看不出该给什么。
    import argparse
    ap = argparse.ArgumentParser(description="离线自检：只读元数据，不碰 latent")
    ap.add_argument("--work", required=True,
                    help="含 captions.jsonl / manifest.jsonl / subject_timing.jsonl 的目录")
    ap.add_argument("--split", default=None, help="train / additional，省略则不筛")
    ap.add_argument("--ref-mode", default="head", choices=["head", "exit"])
    a = ap.parse_args()
    w = Path(a.work)
    # subject_timing.jsonl 只有 ref_mode="exit" 才需要，不能列进必需项。
    # camera_phases.jsonl 只在 manifest 没内嵌 phase 时才需要，所以不列为必需。
    # 真缺了会体现在 no_camera_phase 的丢弃计数上。
    need = ["captions.jsonl", "manifest.jsonl"]
    # exit 模式的额外依赖不在这里查：出画时刻可能内嵌在 manifest 里，
    # 按文件名判断会把自包含 manifest 误报成缺文件。真缺了 OoSWindowIndex 会抛明确的错。
    missing = [f for f in need if not (w / f).is_file()]
    if missing:
        raise SystemExit(f"!! {w} 下缺少: {', '.join(missing)}")
    _selftest(work=a.work, split=a.split, ref_mode=a.ref_mode)
