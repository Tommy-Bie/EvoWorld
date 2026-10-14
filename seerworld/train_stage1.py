#!/usr/bin/env python
"""OoS 训练入口。除了【注入我们自己的滑窗 Dataset】以外，与官方 scripts/train.py 等价。

    accelerate launch --num_processes 8 train_stage1.py oos_lora_r128.yaml

【为什么不能用官方 train.py】trainer._init_dataloader() 在 _dataset 为 None 时会构造
PrecomputedDataset —— 那是【一文件一样本】的，而我们有 29.1 万个滑窗，落盘成 29.1 万个文件
既费磁盘又费 IO。所以在 train() 之前把 _dataset 赋好，_init_dataloader() 里的
`if self._dataset is None` 就会跳过它自己那套。不需要改 trainer 任何一行。

【batch_size】trainer 建 DataLoader 时没传 collate_fn，而文本 embedding 是变长的
（process_captions.py 不做 padding，Gemma tokenizer 也不支持批量分词），所以官方模板都写 1。
本脚本在 batch_size>1 时会把 cw_dataset.collate 接到 trainer 的 DataLoader 上
（patch_dataloader_collate），因此 CW_BATCH 可以调大。batch=1 时不接 —— 默认 collate
本来就是对的，而且它顺便把 num_frames 这类 python int 转成 tensor([v])，
而 _patchify_latent_data 正是按 [0].item() 取的。
"""
import os
import sys
import math
from pathlib import Path

import typer
import yaml

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.trainer import LtxvTrainer

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

# 只在 rank 0 打印。8 个 accelerate 进程各跑一遍 main()，各自建索引各自打印，
# 同样的三行会刷 8 遍。索引本身每个 rank 都要建（各自需要 dataset），省不掉，
# 但打印可以只留一份。
IS_RANK0 = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"


def log0(*a, **k):
    if IS_RANK0:
        print(*a, **k)


def load_yaml_strict(path: Path) -> dict:
    """Load YAML while rejecting duplicate mapping keys.

    PyYAML normally keeps the last occurrence silently. In a training config that can
    hide a stale learning rate, step count, output directory, or guidance value.
    """
    text = path.read_text()
    root = yaml.compose(text)

    def walk(node) -> None:
        if isinstance(node, yaml.MappingNode):
            seen: dict[str, object] = {}
            for key_node, value_node in node.value:
                key = str(getattr(key_node, "value", key_node))
                if key in seen:
                    line = key_node.start_mark.line + 1
                    raise typer.BadParameter(f"duplicate YAML key {key!r} at {path}:{line}")
                seen[key] = key_node
                walk(value_node)
        elif isinstance(node, yaml.SequenceNode):
            for item in node.value:
                walk(item)

    if root is not None:
        walk(root)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"training config root must be a mapping: {path}")
    return loaded


def patch_ddp_num_blocks() -> None:
    """让 DDP 转发 num_blocks，否则 DDP 下第一次 validation 就崩。

    上游 trainer.py:894 把【包装后的】self._transformer 传给 validation runner，
    而 runner 在 validation_runner.py:932/944 访问 transformer.num_blocks。
    DDP 的 __getattr__ 对未注册属性直接抛 AttributeError，于是：
        AttributeError: 'DistributedDataParallel' object has no attribute 'num_blocks'
    transformer.parameters() 和 transformer(...) 在 DDP 上都正常，只有这一个属性没转发。

    为什么不改上游改成 unwrap_model：_run_validation 里明确处理了 FSDP
    （给短 rank 填充 no-save 副本以保持集合通信同步），而 FSDP 下 unwrap 会拿到
    分片参数。无条件 unwrap 会修好 DDP 却弄坏 FSDP。这里只补一个属性转发，
    不改变任何其他行为。
    """
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
    except Exception:
        return
    if not hasattr(DDP, "num_blocks"):
        DDP.num_blocks = property(lambda self: self.module.num_blocks)


def patch_dataloader_collate(target_ds, collate_fn) -> bool:
    """让 trainer 建 DataLoader 时用我们的 collate_fn（仅 batch>1 需要）。

    trainer._init_dataloader 不接受 collate_fn，而文本 embedding 是变长的
    （process_captions.py 不做 padding），所以 batch>1 时默认 collate 会抛 shape 不匹配。

    patch 点是 `ltx_trainer.trainer.DataLoader`：trainer.py 顶部是
    `from torch.utils.data import DataLoader`，那个名字绑在 trainer 模块的命名空间里，
    所以改 torch.utils.data.DataLoader 是没用的（trainer 早已持有旧引用）。

    只对【我们注入的那个 dataset】生效，不影响 trainer 可能建的其他 DataLoader。
    """
    try:
        import ltx_trainer.trainer as T
    except Exception:
        return False
    orig = T.DataLoader

    class _DL(orig):
        def __init__(self, dataset, *a, **k):
            if dataset is target_ds and k.get("collate_fn") is None:
                k["collate_fn"] = collate_fn
            super().__init__(dataset, *a, **k)

    T.DataLoader = _DL
    return True

# 上游 trainer 只记 train/{loss,learning_rate,step_time,global_step} + sigma 分桶损失。
# grad_norm 和 epoch 都没有：trainer.py:235 调 clip_grad_norm_ 但【丢掉了返回值】，
# 而 accelerate 的 clip_grad_norm_ 返回的正是总范数。
#
# 这里用包装而不是改 trainer.py：改上游文件会在每次从 LTX 上游 pull 时产生冲突。
# 代价是包装可能因上游重构而静默失效（指标就那么消失了），所以下面加了一个显式警告 ——
# 把"静默降级"换成"看得见的提示"。
_GRAD = {}


def attach_extra_metrics(trainer, n_samples: int, eff_batch: int):
    acc = trainer._accelerator                     # noqa: SLF001
    orig_clip = acc.clip_grad_norm_

    def clip(*a, **k):
        n = orig_clip(*a, **k)
        try:
            _GRAD["v"] = float(n)                  # DeepSpeed 下会返回 None，故 try
        except (TypeError, ValueError):
            pass
        return n

    acc.clip_grad_norm_ = clip
    orig_log = trainer._log_metrics                # noqa: SLF001
    state = {"n": 0, "warned": False}

    def log(metrics):
        if "train/loss" in metrics:
            state["n"] += 1
            if "v" in _GRAD:
                metrics["train/grad_norm"] = _GRAD["v"]
            elif state["n"] > 20 and not state["warned"]:
                state["warned"] = True
                log0("⚠️ 连续 20 个优化步都没抓到 grad_norm —— clip_grad_norm_ 的包装可能"
                      "因上游改动失效，或 max_grad_norm<=0 让它根本没被调用。", flush=True)
            step = metrics.get("train/global_step", 0)
            # epoch 是推导的：优化步 × 有效 batch / 样本数。trainer 本身没有 epoch 概念
            # （它按步数跑，不按 epoch），所以这个数只是"看过多少遍数据"的直观量。
            if n_samples:
                metrics["train/epoch"] = step * eff_batch / n_samples
        orig_log(metrics)

    trainer._log_metrics = log                     # noqa: SLF001


@app.command()
def main(
    config_path: str = typer.Argument(..., help="YAML 配置路径"),
    # 默认值来自 .env 导出的环境变量。空字符串默认 + 显式检查，
    # 比写死 "work"/"latents" 这类相对路径好 —— 后者在脚本随仓库搬家后会静默指向错的地方。
    work: str = typer.Option(lambda: os.environ.get("CW_WORK", ""), help="三个 jsonl 所在目录"),
    latents: str = typer.Option(lambda: os.environ.get("CW_LATENTS", ""), help="整片 latent 目录"),
    conditions: str = typer.Option(lambda: os.environ.get("CW_CONDITIONS", ""), help="文本 embedding 目录"),
    reference: str = typer.Option(lambda: os.environ.get("CW_REFERENCE", ""), help="reference latent 目录"),
    scene_reference: str = typer.Option(lambda: os.environ.get("CW_SCENE_REFERENCE", ""),
                                        help="scene reference latent 目录（pan peak 单 latent）；"
                                             "空则不启用第二个 reference。目录 basename 必须与 yaml "
                                             "里对应 reference 条件的 latents_dir 一致。"),
    split: str = typer.Option("train"),
    cw_dataset_path: str = typer.Option("", help="cw_dataset.py 所在目录（默认与本脚本同级）"),
    disable_progress_bars: bool = typer.Option(False, "--disable-progress-bars"),
) -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise typer.Exit(f"配置不存在: {cfg_path}")
    for name, val in (("--work", work), ("--latents", latents),
                      ("--conditions", conditions), ("--reference", reference)):
        if not val:
            raise typer.Exit(f"{name} 未给，且对应的 CW_* 环境变量也没设。"
                             f"请先 `source .env` 并确认路径配置。")
    raw_config = load_yaml_strict(cfg_path)
    config = LtxTrainerConfig(**raw_config)

    sys.path.insert(0, cw_dataset_path or str(Path(__file__).resolve().parent))
    from cw_dataset import (
        OoSWindowDataset, PREFIX_LAT, TARGET_LAT, REF_LAT, WINDOW_LAT,
        TEMPORAL_FACTOR, collate,
    )

    if config.validation.interval and config.validation.samples:
        validation_frames = int(config.validation.video_dims[2])
        validation_latents = (validation_frames - 1) // TEMPORAL_FACTOR + 1
        if validation_latents != WINDOW_LAT:
            raise typer.Exit(
                f"validation length mismatch: {validation_frames} frames -> {validation_latents} latents, "
                f"but training window is {WINDOW_LAT} ({PREFIX_LAT} prefix + {TARGET_LAT} target). "
                f"Use {1 + TEMPORAL_FACTOR * (WINDOW_LAT - 1)} frames.")

    iaa_on = int(os.environ.get("CW_USE_IAA", "0") or 0) != 0
    i2v_on = int(os.environ.get("CW_I2V_START", "0") or 0) != 0
    if iaa_on:
        # IAA has stronger structural assumptions than generic flexible training. Validate
        # every one before constructing the 22B model so a typo cannot silently train a
        # different experiment.
        beta = float(os.environ.get("CW_IAA_BETA", "3.0") or 3.0)
        entity_refs = int(os.environ.get("CW_IAA_ENTITY_REFS", "2") or 2)
        expected_refs = 1 if i2v_on else REF_LAT
        if not math.isfinite(beta) or beta < 0:
            raise typer.Exit(f"CW_IAA_BETA must be finite and non-negative, got {beta}")
        if entity_refs != expected_refs:
            raise typer.Exit(
                f"IAA reference count mismatch: {'i2v' if i2v_on else 'v2v'} requires "
                f"CW_IAA_ENTITY_REFS={expected_refs}, got {entity_refs}")
        out_dir = str(raw_config.get("output_dir", ""))
        if "_iaafix" not in Path(out_dir).name:
            raise typer.Exit(
                f"refusing IAA run without _iaafix output tag: {out_dir}. "
                "Regenerate configs with `bash run.sh configs`.")
        if i2v_on != ("_i2v" in Path(out_dir).name):
            raise typer.Exit(
                f"I2V mode/output mismatch: CW_I2V_START={int(i2v_on)}, output_dir={out_dir}")
        lora_cfg = raw_config.get("lora") or {}
        if int(lora_cfg.get("alpha", -1)) != int(lora_cfg.get("rank", -2)):
            raise typer.Exit(
                f"IAA LoRA requires alpha=rank for scale 1.0, got "
                f"rank={lora_cfg.get('rank')} alpha={lora_cfg.get('alpha')}")
        expected_targets = {
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
        }
        actual_targets = set(lora_cfg.get("target_modules") or [])
        if actual_targets != expected_targets:
            raise typer.Exit(
                f"IAA LoRA target_modules mismatch: missing={sorted(expected_targets-actual_targets)}, "
                f"extra={sorted(actual_targets-expected_targets)}")
        if config.training_strategy.audio is not None:
            raise typer.Exit("OoS/IAA is video-only, but the config enables an audio training branch")

    # ★ prefix 长度在两处独立声明：这里的 PREFIX_LAT 决定切多少个 latent 进窗口，
    # yaml 里的 prefix.temporal_boundary 决定 trainer 把前几个当作干净条件。
    # 两者不一致不会报错 —— trainer 只会把窗口的前 N 个当条件，剩下的当 target，
    # 于是"哪些 latent 不计 loss"和我们以为的不一样，而 loss 曲线看起来完全正常。
    prefix_cfgs = [c for c in config.training_strategy.video.conditions
                   if getattr(c, "type", None) == "prefix"]
    if len(prefix_cfgs) != 1:
        raise typer.Exit(f"OoS training requires exactly one prefix condition, got {len(prefix_cfgs)}")
    if float(prefix_cfgs[0].probability) != 1.0:
        raise typer.Exit(
            f"OoS prefix probability must be 1.0, got {prefix_cfgs[0].probability}; "
            "otherwise clean-prefix and target loss roles vary silently.")
    tb = prefix_cfgs[0].temporal_boundary
    if tb != PREFIX_LAT:
        raise typer.Exit(
            f"prefix 长度不一致：yaml temporal_boundary={tb} 而 cw_dataset.PREFIX_LAT={PREFIX_LAT}。"
            f"改成一致再跑 —— 不一致时 trainer 不会报错，只会把 loss 算在错误的 latent 上。")

    # reference 的目录名必须等于 yaml 里 latents_dir：_apply_reference_condition 取的是
    # batch[config.latents_dir]，那个字符串既是目录名也是 batch 的 key。
    # yaml 里可能有多条 reference（主 ref + scene_ref），逐一对齐。
    ref_cfgs = [c for c in config.training_strategy.video.conditions
                if getattr(c, "type", None) == "reference"]
    if iaa_on:
        if len(ref_cfgs) != 1:
            raise typer.Exit(f"IAA requires exactly one entity reference condition, got {len(ref_cfgs)}")
        if float(ref_cfgs[0].probability) != 1.0:
            raise typer.Exit(
                f"IAA reference probability must be 1.0, got {ref_cfgs[0].probability}; "
                "otherwise some batches have no entity tokens while the mask assumes they do.")
    yaml_ref_dirs = [c.latents_dir for c in ref_cfgs]
    cli_ref_dirs = [Path(reference).name] + ([Path(scene_reference).name] if scene_reference else [])
    # 顺序无关：把两侧 sorted 后比对，两边的目录名集合必须一致。
    if sorted(yaml_ref_dirs) != sorted(cli_ref_dirs):
        raise typer.Exit(
            f"reference 条件对不齐：yaml latents_dir={yaml_ref_dirs}，CLI 提供的目录名 "
            f"{cli_ref_dirs}。yaml 里每一条 reference 都要有对应的 CLI 目录同名传入。"
            f"trainer 会用 yaml 的名字去 batch 里取键，取不到就 KeyError，"
            f"而且因为 reference 是按 probability 抽的，可能跑几分钟后才炸。")

    ds = OoSWindowDataset(work=work, latents_dir=latents, conditions_dir=conditions,
                          reference_dir=reference,
                          scene_reference_dir=(scene_reference or None),
                          split=split,
                          oversample_revisit=int(os.environ.get("CW_REVISIT_OVERSAMPLE", "1")),
                          prefix_noise=float(os.environ.get("CW_PREFIX_NOISE", "0")),
                          oversample_i2v_start=int(os.environ.get("CW_I2V_OVERSAMPLE", "1")),
                          i2v_start=(int(os.environ.get("CW_I2V_START", "0") or 0) != 0))
    if float(os.environ.get("CW_PREFIX_NOISE", "0")) > 0:
        log0(f"[oos] prefix noise={os.environ['CW_PREFIX_NOISE']}（前 {4} 个 prefix latent 加噪，缓解 exposure bias）")
    if int(os.environ.get("CW_I2V_START", "0") or 0) != 0:
        log0(f"[oos] I2V-START 开启：reference 用首帧，第一段窗口仅首帧 condition（CW_I2V_OVERSAMPLE={os.environ.get('CW_I2V_OVERSAMPLE','1')}倍）")
    idx = ds.index
    log0(f"[oos] {split} 样本 {len(ds):,}   revisit {sum(1 for s in idx.samples if s['is_revisit']):,}"
          f"   prefix={PREFIX_LAT} target={TARGET_LAT} 窗口={idx.window_lat} latents")
    if idx.dropped:
        log0(f"[oos] 丢弃: {dict(sorted(idx.dropped.items(), key=lambda kv: -kv[1]))}")
    if idx.n_seg_s_fallback:
        log0(f"[oos] {idx.n_seg_s_fallback:,} 条 caption 缺 seg_s，按时长回填（10s→3.375 / 15s→3.75）")
    if len(ds) == 0:
        raise typer.Exit("样本为 0，检查 work/ 下三个 jsonl 与 split 字段")
    if iaa_on:
        n_revisit = sum(1 for s in idx.samples if s["is_revisit"])
        if n_revisit == 0:
            raise typer.Exit("IAA dataset contains zero revisit windows; the attention branch would never train")
        probe = ds[0]
        ref_key = Path(reference).name
        actual_ref_latents = int(probe[ref_key]["latents"].shape[1])
        expected_ref_latents = int(os.environ["CW_IAA_ENTITY_REFS"])
        actual_window_latents = int(probe["video_latents"]["latents"].shape[1])
        video_tensor = probe["video_latents"]["latents"]
        meta_h = int(probe["video_latents"]["height"])
        meta_w = int(probe["video_latents"]["width"])
        tensor_h, tensor_w = map(int, video_tensor.shape[-2:])
        if (meta_h, meta_w) != (tensor_h, tensor_w) or meta_h <= 0 or meta_w <= 0:
            raise typer.Exit(
                f"latent spatial metadata mismatch: metadata={meta_h}x{meta_w}, "
                f"tensor={tensor_h}x{tensor_w}. IAA token ranges would be wrong.")
        if actual_ref_latents != expected_ref_latents:
            raise typer.Exit(
                f"actual reference latent length={actual_ref_latents}, but IAA expects "
                f"{expected_ref_latents} ({ref_key})")
        if actual_window_latents != WINDOW_LAT:
            raise typer.Exit(
                f"actual target window length={actual_window_latents}, expected WINDOW_LAT={WINDOW_LAT}")
        log0(
            f"[IAA/preflight] beta={beta} mode={'i2v' if i2v_on else 'v2v'} "
            f"entity_ref={actual_ref_latents} window={actual_window_latents} grid={meta_h}x{meta_w} "
            f"revisit={n_revisit:,} output={Path(out_dir).name} ✓")

    # ★ 取一个样本，把键名和 strategy 声明的 data_sources 对一遍。
    # 这几个键错了不会在启动时报错 —— 要等 dataloader 吐出第一个 batch、
    # 8 卡都起来、模型都加载完之后才在 trainer.py:368 抛 KeyError。
    # 而键名的真源是 get_data_sources()，不是 PrecomputedDataset 的默认映射
    # （我第一版照后者写成 latent_conditions / text_conditions，就是这么炸的）。
    want = set(config.training_strategy.get_data_sources().values())
    got = set(ds[0].keys()) - {"meta", "idx"}
    if not want.issubset(got):
        raise typer.Exit(
            f"batch 键名不匹配。strategy 需要 {sorted(want)}，"
            f"而 Dataset 给出 {sorted(got)}，缺 {sorted(want - got)}。")
    log0(f"[oos] batch 键: {sorted(got)}  ✓ 与 strategy 声明一致")

    # 必须在 trainer.train() 之前打：DDP 包装发生在 train() 里面，
    # 但补丁是打在 DDP 类上的，所以只要早于包装就行。
    patch_ddp_num_blocks()

    # batch>1 才需要接 collate。batch=1 时默认 collate 是对的（它还顺便把 num_frames
    # 这类 python int 转成 tensor([v])，而 _patchify_latent_data 正是按 [0].item() 取的），
    # 所以不必要地替换掉它反而多一处能出错的地方。
    if config.optimization.batch_size > 1:
        if not patch_dataloader_collate(ds, collate):
            raise typer.Exit("batch_size>1 需要接 collate_fn，但 patch 失败")
        log0(f"[oos] batch_size={config.optimization.batch_size} > 1 → 已接上 cw_dataset.collate")

    trainer = LtxvTrainer(config)
    trainer._dataset = ds        # noqa: SLF001 —— 见文件头，这是唯一的注入点

    if config.wandb.enabled and not os.environ.get("WANDB_API_KEY"):
        raise typer.Exit("wandb.enabled=true 但 WANDB_API_KEY 未设置。"
                         "它在项目环境配置中有默认值，走 `bash run.sh train <rank>` "
                         "会自动带上；直接调本脚本时需要先 `source .env`。\n"
                         "或 wandb.enabled=false 关掉。")
    eff = (config.optimization.batch_size
           * config.optimization.gradient_accumulation_steps
           * int(os.environ.get("WORLD_SIZE", "1")))
    attach_extra_metrics(trainer, len(ds), eff)
    log0(f"[oos] wandb={'on' if config.wandb.enabled else 'off'}"
          f"  有效 batch={eff}  1 epoch≈{len(ds)/max(eff,1):.0f} 步"
          f"  本次 {config.optimization.steps} 步≈{config.optimization.steps*eff/max(len(ds),1):.2f} epoch")
    trainer.train(disable_progress_bars=disable_progress_bars)


if __name__ == "__main__":
    app()
