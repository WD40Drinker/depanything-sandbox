import torch
import onnx
import sys

sys.path.append("depth_anything_repo")

from depth_anything_v2.dpt import DepthAnythingV2


INPUT_SIZE = 224


def load_model():
    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        use_bn=False,
        use_clstoken=False
    )

    # 🔥 IMPORTANT FIX
    model.max_depth = 10.0

    ckpt = torch.load("depth_anything_v2_vits.pth", map_location="cpu")
    model.load_state_dict(ckpt, strict=True)

    model.eval()
    return model


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


def export(model):
    model = Wrapper(model)

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy,
        "depth_anything.onnx",
        input_names=["input"],
        output_names=["depth"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None
    )

    print("✔ exported ONNX")


if __name__ == "__main__":
    model = load_model()
    export(model)