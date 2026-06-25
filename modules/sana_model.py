#  SANA model wrapper around the diffusers pipelines.
#
#  SANA ships in two flavours that share the same diffusers component layout
#  (Gemma2 text encoder + SanaTransformer2DModel + AutoencoderDC "DC-AE"):
#    * SanaPipeline        - the regular, CFG-guided model (~20 steps)
#    * SanaSprintPipeline  - the timestep-distilled "Sprint" model (1-4 steps)
#  Both are auto-detected from the model_index.json `_class_name` field, so a
#  single loader can handle either by deferring to `DiffusionPipeline`.

import os
from typing import Optional, List, Callable

import torch
from diffusers import DiffusionPipeline


#  Map the user-facing dtype string to a torch dtype.
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_device(device: str) -> str:
    """Turn the 'auto' choice into a concrete device for the current machine."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SanaModel:
    """Holds a loaded SANA diffusers pipeline and runs generation for the nodes."""

    def __init__(self):
        self.pipe: Optional[DiffusionPipeline] = None
        self.model_path: Optional[str] = None
        self.device: str = "cpu"
        self.is_sprint: bool = False

    def load(self, model_path: str, device: str, dtype: str) -> None:
        resolved_device = resolve_device(device)
        torch_dtype = DTYPE_MAP[dtype]

        #  `DiffusionPipeline` reads model_index.json and instantiates the right
        #  subclass (SanaPipeline / SanaSprintPipeline) for us.
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        )

        #  The Sprint scheduler is SCM and is CFG-distilled; we use this flag to
        #  decide which generation kwargs are meaningful.
        self.is_sprint = type(pipe).__name__ == "SanaSprintPipeline"

        #  SANA recommends keeping the text encoder + transformer in the chosen
        #  precision but the DC-AE VAE is most stable in float32. diffusers
        #  already handles this internally for fp16/bf16 loads, so we just move
        #  the whole pipeline to the device.
        pipe = pipe.to(resolved_device)

        self.pipe = pipe
        self.model_path = model_path
        self.device = resolved_device

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        num_images: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List["Image.Image"]:  # noqa: F821 - PIL imported lazily by caller
        if self.pipe is None:
            raise RuntimeError("SANA pipeline is not loaded. Run the loader node first.")

        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        #  Bridge diffusers' step callback to ComfyUI's progress bar.
        def _on_step_end(pipe, step, timestep, callback_kwargs):
            if progress_callback is not None:
                progress_callback(step + 1, steps)
            return callback_kwargs

        kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
            output_type="pil",
            callback_on_step_end=_on_step_end,
        )

        #  The Sprint pipeline has no negative_prompt argument (CFG distilled),
        #  so only pass it for the regular SANA pipeline.
        if not self.is_sprint:
            kwargs["negative_prompt"] = negative_prompt or ""

        result = self.pipe(**kwargs)
        return result.images
