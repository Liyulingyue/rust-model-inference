# Astronex-World converter

Export the pinned Wan2.2 TI2V-5B checkpoint without dtype conversion:

```bash
.venv/bin/python -m tools.converter.astronex_world.convert_astronex_world \
  models/Astronex-World --out-dir models/Astronex-World
```

The exporter writes `Astronex-World-BF16.gguf` and
`mmproj-Astronex-World.gguf`. It validates the pinned 898-tensor denoiser,
242-tensor UMT5-XXL encoder, and 196-tensor F32 Wan VAE contract before
publishing the pair atomically.
