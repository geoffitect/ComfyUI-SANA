#  ComfyUI-SANA nodes
#
#  A small node pack that runs NVIDIA / Efficient-Large-Model's SANA family
#  (regular SANA and the distilled SANA-Sprint) through the diffusers pipelines.
#
#  Two nodes, kept idiomatic to ComfyUI:
#    * SanaModelLoader  - loads a diffusers-format SANA folder -> SANA_MODEL
#    * SanaGenerate     - runs the pipeline -> IMAGE
import os

import numpy as np
import torch

#  ComfyUI modules
import folder_paths
from comfy.utils import ProgressBar

#  Package modules
from .modules.sana_model import SanaModel, DTYPE_MAP


def _list_sana_models():
    """Find diffusers-format models under every registered `diffusers` path.

    A folder qualifies if it contains a model_index.json (the diffusers pipeline
    manifest). We don't filter on class name here so any SANA variant the user
    drops in (or symlinks from the HF cache) shows up automatically.
    """
    found = []
    for base in folder_paths.get_folder_paths("diffusers"):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if os.path.isfile(os.path.join(path, "model_index.json")):
                found.append(name)
    return found


class SanaModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        models = _list_sana_models()
        return {
            "required": {
                #  Dropdown of diffusers folders under ComfyUI/models/diffusers.
                "model": (models if models else ["(no diffusers models found)"],),
                "device": (["auto", "mps", "cuda", "cpu"],),
                "dtype": (list(DTYPE_MAP.keys()), {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("SANA_MODEL",)
    RETURN_NAMES = ("sana_model",)
    FUNCTION = "load_model"
    CATEGORY = "SANA"

    def load_model(self, model: str, device: str, dtype: str):
        base_candidates = folder_paths.get_folder_paths("diffusers")
        model_path = None
        for base in base_candidates:
            candidate = os.path.join(base, model)
            if os.path.isdir(candidate):
                model_path = candidate
                break
        if model_path is None:
            raise FileNotFoundError(
                f"Could not find diffusers model '{model}' under {base_candidates}"
            )

        sana = SanaModel()
        sana.load(model_path, device, dtype)
        return (sana,)


class SanaGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sana_model": ("SANA_MODEL",),
                "prompt": ("STRING", {"default": "a cyberpunk cat, neon city", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                #  Sprint wants 1-4 steps; regular SANA ~20. Default suits Sprint.
                "steps": ("INT", {"default": 2, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "SANA"

    def generate(
        self,
        sana_model: SanaModel,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        batch_size: int,
    ):
        comfy_pbar = ProgressBar(steps)

        def _progress(step, total):
            comfy_pbar.update_absolute(step, total)

        images = sana_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=batch_size,
            progress_callback=_progress,
        )

        #  Convert PIL images -> ComfyUI IMAGE tensor: float32, NHWC, range [0, 1].
        tensors = []
        for img in images:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
        image_batch = torch.stack(tensors, dim=0)
        return (image_batch,)
