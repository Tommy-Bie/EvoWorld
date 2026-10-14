# Training environment

The public installation is managed with [uv](https://docs.astral.sh/uv/).
The following core versions were recorded from the original training
environment on 2026-10-13:

| Component | Version |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.13.0+cu132 |
| TorchVision | 0.28.0 (the CUDA wheel may report 0.28.0+cu132) |
| TorchAudio | 2.11.0+cu132 |
| CUDA build | 13.2 |
| Accelerate | 1.14.0 |
| Transformers | 5.14.1 |
| PEFT | 0.20.0 |
| NATTEN | 0.21.7+torch2130cu132 |
| Safetensors | 0.8.0 |

`diffusers` was not installed. SeerWorld uses the local LTX workspace packages
under `packages/` instead of Diffusers pipelines.

Create a clean environment from the repository root with:

```bash
uv sync --extra natten --python 3.10.20
uv run python scripts/check_environment.py
```

The CUDA 13.2 wheels are Linux-only and require a compatible NVIDIA driver.
The environment check validates package/build versions and CUDA availability;
it does not by itself reproduce an end-to-end training run.
