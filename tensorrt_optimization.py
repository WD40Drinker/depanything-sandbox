"""
Depth Anything V2 - TensorRT Optimization Pipeline
Optimized for local .pth checkpoints (ViT-S/B/L encoders)

Requirements:
    pip install torch torchvision onnx onnxruntime pycuda
    pip install tensorrt==8.6.1  (from NVIDIA wheel)
    conda install cudnn=8.9 -c conda-forge
"""

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

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192,  384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384,  768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ──────────────────────────────────────────────
# 1. Load model
# ──────────────────────────────────────────────

def load_model(pth_path: str, encoder: str = 'vits', max_depth: int = 10, device: str = 'cuda'):
    sys.path.append('metric_depth')
    from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

    print(f"[1/4] Loading model: {pth_path}  encoder={encoder}")
    model = DepthAnythingV2(**{**MODEL_CONFIGS[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval().to(device)
    print(f"      Loaded on {device}")
    return model


# ──────────────────────────────────────────────
# 2. Export to ONNX
# ──────────────────────────────────────────────

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)   # returns (B, H, W)
    
class CleanWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)


def export_onnx(model, onnx_path: str, input_hw: tuple = (518, 518), opset: int = 17):
    print(f"\n[2/4] Exporting ONNX → {onnx_path}")

    wrapper = CleanWrapper(model)
    device  = next(model.parameters()).device
    dummy   = torch.randn(1, 3, *input_hw, dtype=torch.float32, device=device)

    torch.onnx.export(
        wrapper,
        (dummy,),
        onnx_path,
        input_names=["input"],
        output_names=["depth"],
        opset_version=12,
        do_constant_folding=True,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,  # IMPORTANT
    )

    onnx.checker.check_model(onnx.load(onnx_path))
    size_mb = Path(onnx_path).stat().st_size / 1e6
    print(f"      Validated  ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────
# 3. Build TensorRT engine
# ──────────────────────────────────────────────

def build_engine(onnx_path: str,
                 engine_path: str,
                 fp16: bool = True,
                 min_hw: tuple = (256, 256),
                 opt_hw: tuple = (518, 518),
                 max_hw: tuple = (1024, 1024),
                 workspace_gb: int = 1):

    print(f"\n[3/4] Building TensorRT engine → {engine_path}")
    print(f"      Precision : {'FP16' if fp16 else 'FP32'}")

    builder = trt.Builder(TRT_LOGGER)
    flags   = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser  = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"      Parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("      FP16 enabled")


    print("      Compiling… (this may take several minutes)")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    elapsed = time.time() - t0
    size_mb = Path(engine_path).stat().st_size / 1e6
    print(f"      Done in {elapsed:.1f}s  ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────
# 4. TensorRT inference engine
# ──────────────────────────────────────────────

class TRTEngine:
    def __init__(self, engine_path: str):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()
        self._setup_buffers()
        print(f"[TRTEngine] Loaded {engine_path}")

    def _setup_buffers(self):
        self.tensors = {}
        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode  = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)

            # Use max profile shape for buffer allocation
            if any(d < 0 for d in shape):
                max_shape = self.engine.get_tensor_profile_shape(name, 0)[2]
                size = 1
                for d in max_shape:
                    size *= int(d)
            else:
                size = 1
                for d in shape:
                    size *= int(d)

            host_buf   = cuda.pagelocked_empty(size, dtype)
            device_buf = cuda.mem_alloc(host_buf.nbytes)

            self.tensors[name] = {
                "host":   host_buf,
                "device": device_buf,
                "mode":   mode,
                "dtype":  dtype,
            }
            self.context.set_tensor_address(name, int(device_buf))

    def infer(self, x: np.ndarray) -> np.ndarray:
        input_name = next(n for n,t in self.tensors.items() if t["mode"] == trt.TensorIOMode.INPUT)
        output_name = next(n for n,t in self.tensors.items() if t["mode"] == trt.TensorIOMode.OUTPUT)

        self.context.set_input_shape(input_name, x.shape)

        out_shape = tuple(int(d) for d in self.context.get_tensor_shape(output_name))
        out_size = int(np.prod(out_shape))

        if self.tensors[output_name]["host"].size < out_size:
            self.tensors[output_name]["host"] = cuda.pagelocked_empty(out_size, self.tensors[output_name]["dtype"])
            self.tensors[output_name]["device"] = cuda.mem_alloc(self.tensors[output_name]["host"].nbytes)
            self.context.set_tensor_address(output_name, int(self.tensors[output_name]["device"]))

        flat = np.ascontiguousarray(x.astype(self.tensors[input_name]["dtype"])).ravel()
        self.tensors[input_name]["host"][:flat.size] = flat

        cuda.memcpy_htod_async(self.tensors[input_name]["device"], self.tensors[input_name]["host"], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.tensors[output_name]["host"], self.tensors[output_name]["device"], self.stream)
        self.stream.synchronize()

        return self.tensors[output_name]["host"][:out_size].reshape(out_shape).copy()

    def __del__(self):
        try:
            for t in self.tensors.values():
                t["device"].free()
        except Exception:
            pass


# ──────────────────────────────────────────────
# 5. Benchmark
# ──────────────────────────────────────────────

def benchmark(engine: TRTEngine,
              input_hw: tuple = (320, 320),
              batch_size: int = 1,
              warmup: int = 10,
              runs: int = 100):

    dummy = np.random.randn(batch_size, 3, *input_hw).astype(np.float32)
    print(f"\n[Benchmark]  batch={batch_size}  hw={input_hw}  runs={runs}")

    for _ in range(warmup):
        engine.infer(dummy)

    t0 = time.perf_counter()
    for _ in range(runs):
        engine.infer(dummy)
    elapsed = time.perf_counter() - t0

    mean_ms = elapsed / runs * 1000
    fps     = batch_size * 1000 / mean_ms
    print(f"  Latency   : {mean_ms:.2f} ms")
    print(f"  Throughput: {fps:.1f} FPS")
    return mean_ms, fps


# ──────────────────────────────────────────────
# 6. Pipeline
# ──────────────────────────────────────────────

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    onnx_path   = os.path.join(args.output_dir, "depth_anything.onnx")
    engine_path = os.path.join(args.output_dir, "depth_anything.trt")

    # Export ONNX
    if not os.path.exists(onnx_path) or args.force:
        model = load_model(args.pth_path, args.encoder, args.max_depth, device)
        export_onnx(model, onnx_path, (args.input_h, args.input_w), args.opset)
    else:
        print(f"[2/4] ONNX already exists, skipping  ({onnx_path})")

    # Build TRT engine
    if not os.path.exists(engine_path) or args.force:
        build_engine(onnx_path, engine_path,
                     fp16=args.fp16,
                     opt_hw=(args.input_h, args.input_w),
                     workspace_gb=args.workspace_gb)
    else:
        print(f"[3/4] Engine already exists, skipping  ({engine_path})")

    # Load and run
    print(f"\n[4/4] Loading engine for inference")
    engine = TRTEngine(engine_path)

    if args.benchmark:
        benchmark(engine, (args.input_h, args.input_w), args.batch_size)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Depth Anything V2 TensorRT optimizer")

    p.add_argument("--pth-path",     required=True,        help="Path to .pth checkpoint")
    p.add_argument("--encoder",      default="vits",       choices=["vits", "vitb", "vitl"])
    p.add_argument("--max-depth",    type=int, default=10)
    p.add_argument("--input-h",      type=int, default=518)
    p.add_argument("--input-w",      type=int, default=518)
    p.add_argument("--opset",        type=int, default=17)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--workspace-gb", type=int, default=4)
    p.add_argument("--batch-size",   type=int, default=1)
    p.add_argument("--output-dir",   default="./trt_output")
    p.add_argument("--benchmark",    action="store_true")
    p.add_argument("--force",        action="store_true",  help="Re-export and rebuild even if files exist")

    args = p.parse_args()
    run(args)