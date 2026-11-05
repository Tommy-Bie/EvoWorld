# Training environment


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


Create a clean environment from the repository root with:

```bash
uv sync --extra natten --python 3.10.20
uv run python scripts/check_environment.py
```

The environment check validates package/build versions and CUDA availability.
