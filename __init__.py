from .nodes import SanaModelLoader, SanaGenerate


#  Map node classes to the unique keys ComfyUI stores in workflows.
NODE_CLASS_MAPPINGS = {
    "SanaModelLoader": SanaModelLoader,
    "SanaGenerate": SanaGenerate,
}

#  Friendly names shown in the node search / title bar.
NODE_DISPLAY_NAME_MAPPINGS = {
    "SanaModelLoader": "SANA Model Loader",
    "SanaGenerate": "SANA Generate",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
