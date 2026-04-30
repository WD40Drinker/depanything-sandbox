import torch
import onnx
import cv2
import sys

# ---- IMPORT YOUR MODEL ----
sys.path.append("path_to_depth_anything_repo")

from depth_anything_v2.dpt import DepthAnythingV2


# -----------------------------
# CONFIG (IMPORTANT FOR NANO)
# -----------------------------
ENCODER = "vits"          # MUST be vits for Nano (vitb/vitl are too heavy)
INPUT_SIZE = 224          # MUST be divisible by 14
OPSET = 11                # CRITICAL for Jetson Nano


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    model = DepthAnythingV2(
        encoder=ENCODER,
        features=64,
        out_channels=[48, 96, 192, 384],
        max_depth=10.0
    )

    checkpoint = torch.load("depth_anything_v2_vits.pth", map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)

    model.eval()
    return model


# -----------------------------
# WRAPPER (IMPORTANT)
# -----------------------------
class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # return ONLY depth map
        return self.model(x)


# -----------------------------
# EXPORT FUNCTION
# -----------------------------
def export_onnx(model, out_path="depth_anything.onnx"):
    model = ONNXWrapper(model)
    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    print("[INFO] Exporting ONNX...")

    torch.onnx.export(
        model,
        dummy,
        out_path,

        input_names=["input"],
        output_names=["depth"],

        opset_version=OPSET,

        # 🔥 CRITICAL FOR JETSON
        dynamic_axes=None,

        do_constant_folding=True,

        export_params=True
    )

    # validate
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    print(f"[OK] ONNX saved → {out_path}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    model = load_model()
    export_onnx(model, "depth_anything_vits.onnx")