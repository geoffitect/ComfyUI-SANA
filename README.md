# ComfyUI-SANA

ComfyUI nodes for the [SANA](https://github.com/NVlabs/Sana) text-to-image family,
run through 🤗 `diffusers`. Supports both the regular **SANA** models and the
distilled **SANA-Sprint** models (1–4 step generation).

SANA is small and fast: the 0.6B Sprint model generates a 1024×1024 image in a
couple of steps and runs comfortably on Apple Silicon (MPS), CUDA, or CPU.

## Nodes

| Node | Output | Description |
| --- | --- | --- |
| **SANA Model Loader** | `SANA_MODEL` | Loads a diffusers-format SANA folder from `ComfyUI/models/diffusers`. Auto-detects regular vs. Sprint. Pick device (`auto`/`mps`/`cuda`/`cpu`) and dtype (`bfloat16`/`float16`/`float32`). |
| **SANA Generate** | `IMAGE` | Runs the pipeline. Prompt, negative prompt (regular only), width/height, steps, guidance scale, seed, batch size. |

Suggested settings:

- **SANA-Sprint**: `steps` 1–4 (2 is a good default), `guidance_scale` ~4.5. The
  Sprint scheduler is CFG-distilled, so the `negative_prompt` field is ignored.
- **Regular SANA**: `steps` ~20, `guidance_scale` ~4.5, `negative_prompt` honored.

## Installation

Clone into your `custom_nodes` directory and install requirements into the same
Python environment ComfyUI runs in:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gtaghon/ComfyUI-SANA
cd ComfyUI-SANA
pip install -r requirements.txt
```

Requirements: `diffusers>=0.33.0`, `transformers>=4.49.0`, `accelerate`,
`sentencepiece`.

## Models

The loader lists any folder under `ComfyUI/models/diffusers` that contains a
`model_index.json` (the diffusers pipeline manifest).

### Using models already in your Hugging Face cache

If you already have a SANA diffusers model cached (e.g. from a prior
`diffusers` download), symlink it into ComfyUI instead of re-downloading:

```bash
# Find the snapshot directory in the HF cache, then link it in:
SNAP=~/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_Sprint_0.6B_1024px_diffusers/snapshots/*
ln -s $SNAP ~/ComfyUI/models/diffusers/Sana_Sprint_0.6B_1024px_diffusers
```

It then appears in the loader's dropdown.

> **Note on model format:** only **diffusers-format** repos work (those with a
> `model_index.json` plus `transformer/`, `text_encoder/`, `vae/`, etc.). The
> original SANA `.pth` checkpoints (e.g. `Sana_600M_512px_MultiLing.pth`) are
> *not* diffusers-format and won't load directly — use the corresponding
> `..._diffusers` repo from Hugging Face.

### Downloading

To fetch one fresh, download any SANA diffusers repo into
`ComfyUI/models/diffusers/<name>`, e.g.
`Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` or
`Efficient-Large-Model/Sana_600M_512px_diffusers`.

## Example workflow

See [`workflows/sana-sprint-txt2img.json`](workflows/sana-sprint-txt2img.json):
**SANA Model Loader → SANA Generate → Save Image**.

## License

The node code is provided as-is. SANA model weights are subject to their own
license from NVIDIA / Efficient-Large-Model.
