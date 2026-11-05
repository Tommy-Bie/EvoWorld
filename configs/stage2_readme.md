# Stage 2 Configuration

The Stage 2 configuration is generated dynamically from `configs/stage1.yaml`
when running:

```bash
bash scripts/train_stage2.sh
```

Set `EW_STAGE1_LORA` to the Stage 1 checkpoint before launching training. The
generated configuration is written to `configs/generated/stage2.yaml`.
