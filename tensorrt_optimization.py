import os
import torch
import torch.nn as nn
import onnx
from pathlib import Path


# ─────────────────────────────
# MODEL WRAPPER
# ─────────────────────────────

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Depth Anything returns (B, H, W) or (B, 1, H, W)
        return self.model(x)


# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────

def load_model(pth_path, encoder="vits", max_depth=10):
    import sys
    sys.path.append("metric_depth")

    from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder=encoder,
        max_depth=max_depth
    )

    state = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    return model


# ─────────────────────────────
# EXPORT ONNX
# ─────────────────────────────

def export_onnx(model, onnx_path, input_size=(280, 280)):
    model = ModelWrapper(model).eval()

    dummy = torch.randn(1, 3, *input_size)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["depth"],
        opset_version=11,          # IMPORTANT for Jetson Nano
        do_constant_folding=True,
        export_params=True,
        dynamic_axes=None,         # FORCE STATIC GRAPH
    )

    # Validate ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print(f"✔ ONNX exported: {onnx_path}")
    print(f"✔ Size: {Path(onnx_path).stat().st_size / 1e6:.2f} MB")


# ─────────────────────────────
# MAIN
# ─────────────────────────────

if __name__ == "__main__":
    pth_path = "model.pth"              # CHANGE THIS
    onnx_path = "depth.onnx"            # OUTPUT FILE

    model = load_model(
        pth_path=pth_path,
        encoder="vits",
        max_depth=10
    )

    export_onnx(
        model,
        onnx_path,
        input_size=(280, 280)   # IMPORTANT for Jetson Nano stability
    )