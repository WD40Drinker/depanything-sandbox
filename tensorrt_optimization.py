import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa


# ─────────────────────────────
# CONFIG
# ─────────────────────────────

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


# ─────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────

def load_model(pth_path, encoder, max_depth, device):
    sys.path.append('metric_depth')
    from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        **MODEL_CONFIGS[encoder],
        max_depth=max_depth
    )

    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval().to(device)
    return model


# ─────────────────────────────
# 2. ONNX EXPORT (FIXED)
# ─────────────────────────────

class Wrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


def export_onnx(model, onnx_path, input_hw=(280, 280)):
    model.eval()

    wrapper = Wrapper(model).eval()

    dummy = torch.randn(1, 3, *input_hw).cuda()

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["depth"],
        opset_version=11,              # IMPORTANT for Nano
        do_constant_folding=True,
        dynamic_axes=None,             # STATIC ONLY
    )

    onnx.checker.check_model(onnx.load(onnx_path))
    print("✔ ONNX exported:", onnx_path)


# ─────────────────────────────
# 3. TENSORRT BUILD (FIXED)
# ─────────────────────────────

def build_engine(onnx_path, engine_path, fp16=True):
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ ONNX parse failed:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError()

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print("🔧 Building TensorRT engine...")

    t0 = time.time()
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(engine)

    print(f"✔ Engine built in {time.time() - t0:.2f}s")


# ─────────────────────────────
# 4. PIPELINE
# ─────────────────────────────

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    onnx_path = os.path.join(args.output_dir, "depth.onnx")
    engine_path = os.path.join(args.output_dir, "depth.trt")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FORCE CLEAN SIZE (IMPORTANT FOR VI T PATCH SIZE)
    args.input_h = 280
    args.input_w = 280

    # STEP 1: ONNX
    if not os.path.exists(onnx_path) or args.force:
        model = load_model(args.pth_path, args.encoder, args.max_depth, device)
        export_onnx(model, onnx_path, (280, 280))

    # STEP 2: TRT
    if not os.path.exists(engine_path) or args.force:
        build_engine(onnx_path, engine_path, fp16=args.fp16)

    print("✔ DONE")


# ─────────────────────────────
# CLI
# ─────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--pth-path", required=True)
    p.add_argument("--encoder", default="vits")
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--output-dir", default="./trt_output")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--force", action="store_true")

    args = p.parse_args()
    run(args)