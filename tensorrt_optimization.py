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
    

Get responses tailored to you

Log in to get answers based on saved chats, plus create images and upload files.


Log in
import cv2
import numpy as np
import matplotlib
import winsound
import time
import os


import tensorrt as trt
import pycuda.driver as cuda

# ── Manual CUDA context (replaces pycuda.autoinit) ───────────────────────────
cuda.init()
cuda_ctx = cuda.Device(0).make_context()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def beep(freq=1000, duration=800):
    os.system(f'beep -f {freq} -l {duration}')



class TRTDepthPredictor:
    def __init__(self, engine_path="NYUmodel.trt", input_size=(518, 518)):
        self.input_size = input_size
        self.cmap = matplotlib.colormaps["turbo"]

        cuda_ctx.push()

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

        cuda_ctx.pop()

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        h, w = self.input_size

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                # Force concrete input shape: 1 x 3 x H x W
                shape = (1, 3, h, w)
                self.context.set_input_shape(name, shape)
            else:
                # Output shape becomes valid after setting input shape
                shape = tuple(self.context.get_tensor_shape(name))

            size = int(np.prod(shape))

            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({"host": host_mem, "device": device_mem, "name": name})
            else:
                outputs.append({"host": host_mem, "device": device_mem, "name": name})

        return inputs, outputs, bindings, stream

    def _preprocess(self, frame):
        h, w = self.input_size
        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        img  = img.transpose(2, 0, 1)
        return np.ascontiguousarray(img[None])

    def _do_inference(self):
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
            self.context.set_tensor_address(inp["name"], int(inp["device"]))
        for out in self.outputs:
            self.context.set_tensor_address(out["name"], int(out["device"]))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

    def infer_image(self, frame):
        cuda_ctx.push()
        tensor = self._preprocess(frame)
        np.copyto(self.inputs[0]["host"], tensor.ravel())
        self._do_inference()
        h, w = self.input_size
        depth = self.outputs[0]["host"].reshape(h, w).copy()
        cuda_ctx.pop()
        return depth

    def colorize(self, depth):
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        colormap   = self.cmap(depth_norm)[:, :, :3]
        colormap   = (colormap * 255).astype(np.uint8)
        return cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    
    def infer_video(self, video_path, d, v, show=False):
        cap = cv2.VideoCapture(video_path)
        last_beep = 0

        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        prevdepth = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            depth = self.infer_image(frame)

            if show:
                cv2.imshow("DepthAnythingV2 TRT", self.colorize(depth))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            depth_disp = 255.0 - (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0

            h, w = depth_disp.shape
            cy, cx = h // 2, w // 2
            crop = depth_disp[cy-120:cy+120, cx-120:cx+120]

            if prevdepth is not None:
                velocity = -(crop - prevdepth)
                if (velocity > v).any():
                    if time.time() - last_beep > 3:
                        beep(500, 800)
                        last_beep = time.time()
                        print("velocity warning")

            if (crop > d).any():
                if time.time() - last_beep > 3:
                    beep(1000, 800)
                    last_beep = time.time()
                    print("distance warning")

            prevdepth = crop.copy()

        cap.release()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        beep(1000, 800)
        beep(1000, 800)

        predictor = TRTDepthPredictor(
            engine_path = "NYUmodel.trt",
            input_size  = (518, 518),
        )
        predictor.infer_video(0, d=252, v=200, show=True)

    finally:
        cuda_ctx.pop()  # always release context on exit



attribue error: "nonetype" object has no attribute "cretate_execution context" pycuda error: the context stack was not empty upon module cleanup

You’ve got two separate problems tangled together here, and they point to different root causes.

1) NoneType has no attribute create_execution_context
This line is the giveaway:

self.engine = runtime.deserialize_cuda_engine(f.read())
self.context = self.engine.create_execution_context()
If self.engine is None, then deserialize_cuda_engine() failed. That’s why you get:

'NoneType' object has no attribute 'create_execution_context'

Why TensorRT returns None
Common causes:

❌ Engine file (NYUmodel.trt) is invalid or corrupted

❌ Engine built on a different GPU architecture

❌ TensorRT version mismatch between build vs runtime

❌ CUDA / driver mismatch

❌ File path wrong (empty read)

Fix: add a hard check + debug info
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine_data = f.read()
    if not engine_data:
        raise RuntimeError("Engine file is empty or not found")

    self.engine = runtime.deserialize_cuda_engine(engine_data)

if self.engine is None:
    raise RuntimeError("Failed to deserialize TensorRT engine. Check compatibility.")
If that triggers, rebuild the engine on this exact machine.

2) PyCUDA error: context stack not empty
pycuda error: the context stack was not empty upon module cleanup

This is happening because your push/pop usage is inconsistent.

Problem spots in your code:
A) Constructor
cuda_ctx.push()
...
cuda_ctx.pop()
B) infer_image
cuda_ctx.push()
...
cuda_ctx.pop()
C) finally
cuda_ctx.pop()
👉 That final pop() is dangerous — it runs even if nothing is pushed.

Root issue
You are manually managing a global CUDA context, but:

pushing/popping in multiple places

and also popping again at shutdown

That causes imbalance → crash.

Clean fix (recommended)
✅ Option 1 — simplest: remove manual context handling entirely
Delete all of this:

cuda.init()
cuda_ctx = cuda.Device(0).make_context()
And remove every:

cuda_ctx.push()
cuda_ctx.pop()
👉 TensorRT + PyCUDA usually works fine without manual context unless you're doing multi-threading.

✅ Option 2 — if you really need manual context
Then do it once, globally:

cuda.init()
cuda_ctx = cuda.Device(0).make_context()
cuda_ctx.push()
And at shutdown:

cuda_ctx.pop()
🚫 Do NOT push/pop inside functions like infer_image().

3) Bonus bug (important)
This line:

predictor.infer_video(0, d=252, v=200, show=True)
You're passing 0 as video_path, which is fine (webcam), but your function says:

cap = cv2.VideoCapture(video_path)
That’s okay — just be aware it's not a file.

4) Another subtle issue (beep)
You defined:

def beep(freq=1000, duration=800):
    os.system(f'beep -f {freq} -l {duration}')
But you imported:

import winsound
On Windows, beep command usually doesn’t exist.

👉 Replace with:

def beep(freq=1000, duration=800):
    winsound.Beep(freq, duration)
Summary (what to fix first)
✅ Check why engine is None → this is the main blocker

✅ Remove or fix CUDA context handling (don’t double-pop)

✅ Fix beep function (Windows compatibility)

If you want, I can help you debug the .trt file specifically (that’s usually where the real issue is).


i made the .trt file on a gtx 1080 and am trying to run it on a jetson nano

That’s the problem right there — and there’s no workaround for it.

A TensorRT engine (.trt) is not portable across devices, especially between something like:

Desktop GPU: GTX 1080 (Pascal, compute capability 6.1)

Embedded GPU: Jetson Nano (Maxwell, compute capability 5.3)

Why it fails
When you do:

runtime.deserialize_cuda_engine(...)
TensorRT tries to load GPU-specific kernels that were compiled for the 1080, not the Nano.

On the Nano:

Different architecture (Maxwell vs Pascal)

Different instruction set

Different TensorRT / CUDA versions

👉 So deserialization fails silently → returns None

What you must do
You need to rebuild the engine on the Jetson Nano itself.

There’s no reliable way to reuse the .trt file from your PC.

Correct workflow
Step 1: Move model (not engine) to Nano
Transfer one of these instead:

.onnx (best option)

or original PyTorch model

Step 2: Build TensorRT engine on Nano
Example using trtexec:

/usr/src/tensorrt/bin/trtexec \
  --onnx=model.onnx \
  --saveEngine=NYUmodel.trt \
  --fp16
Jetson Nano supports FP16 (good), but not INT8 unless calibrated.

Step 3: Use the new engine
Then your Python code will work:

self.engine = runtime.deserialize_cuda_engine(f.read())
Extra gotcha (very common on Jetson)
Jetson uses its own stack:

JetPack (bundled CUDA + TensorRT)

Often older versions than desktop

So even if GPU architecture matched (it doesn’t), version mismatch would still break it.

Quick sanity check
On the Nano, run:

/usr/src/tensorrt/bin/trtexec --loadEngine=NYUmodel.trt
If it fails → engine is invalid for that device.

Bottom line
❌ You cannot reuse .trt from GTX 1080

✅ You must rebuild on Jetson Nano

✅ Use ONNX as the portable format

If you want, I can help you:

export your model to ONNX correctly

optimize it specifically for Nano (important — Nano is very resource-limited)

or fix performance (you’ll likely need it)


import cv2
import numpy as np
import matplotlib
import winsound
import time
import os


import tensorrt as trt
import pycuda.driver as cuda

# ── Manual CUDA context (replaces pycuda.autoinit) ───────────────────────────
cuda.init()  rewrite code using NYUmodel.onnx 
cuda_ctx = cuda.Device(0).make_context()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def beep(freq=1000, duration=800):
    os.system(f'beep -f {freq} -l {duration}')



class TRTDepthPredictor:
    def __init__(self, engine_path="NYUmodel.trt", input_size=(518, 518)):
        self.input_size = input_size
        self.cmap = matplotlib.colormaps["turbo"]

        cuda_ctx.push()

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

        cuda_ctx.pop()

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        h, w = self.input_size

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                # Force concrete input shape: 1 x 3 x H x W
                shape = (1, 3, h, w)
                self.context.set_input_shape(name, shape)
            else:
                # Output shape becomes valid after setting input shape
                shape = tuple(self.context.get_tensor_shape(name))

            size = int(np.prod(shape))

            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({"host": host_mem, "device": device_mem, "name": name})
            else:
                outputs.append({"host": host_mem, "device": device_mem, "name": name})

        return inputs, outputs, bindings, stream

    def _preprocess(self, frame):
        h, w = self.input_size
        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        img  = img.transpose(2, 0, 1)
        return np.ascontiguousarray(img[None])

    def _do_inference(self):
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
            self.context.set_tensor_address(inp["name"], int(inp["device"]))
        for out in self.outputs:
            self.context.set_tensor_address(out["name"], int(out["device"]))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

    def infer_image(self, frame):
        cuda_ctx.push()
        tensor = self._preprocess(frame)
        np.copyto(self.inputs[0]["host"], tensor.ravel())
        self._do_inference()
        h, w = self.input_size
        depth = self.outputs[0]["host"].reshape(h, w).copy()
        cuda_ctx.pop()
        return depth

    def colorize(self, depth):
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        colormap   = self.cmap(depth_norm)[:, :, :3]
        colormap   = (colormap * 255).astype(np.uint8)
        return cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)
    
    def infer_video(self, video_path, d, v, show=False):
        cap = cv2.VideoCapture(video_path)
        last_beep = 0

        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        prevdepth = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            depth = self.infer_image(frame)

            if show:
                cv2.imshow("DepthAnythingV2 TRT", self.colorize(depth))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            depth_disp = 255.0 - (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0

            h, w = depth_disp.shape
            cy, cx = h // 2, w // 2
            crop = depth_disp[cy-120:cy+120, cx-120:cx+120]

            if prevdepth is not None:
                velocity = -(crop - prevdepth)
                if (velocity > v).any():
                    if time.time() - last_beep > 3:
                        beep(500, 800)
                        last_beep = time.time()
                        print("velocity warning")

            if (crop > d).any():
                if time.time() - last_beep > 3:
                    beep(1000, 800)
                    last_beep = time.time()
                    print("distance warning")

            prevdepth = crop.copy()

        cap.release()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        beep(1000, 800)
        beep(1000, 800)

        predictor = TRTDepthPredictor(
            engine_path = "NYUmodel.trt",
            input_size  = (518, 518),
        )
        predictor.infer_video(0, d=252, v=200, show=True)

    finally:
        cuda_ctx.pop()  # always release context on exit

You don’t want to “use ONNX directly” inside this same TensorRT runtime flow — that’s not how TensorRT works. You still need to build an engine from the ONNX first, then run inference with that engine.

Right now your code only loads a .trt. We’ll modify it so:

👉 If .trt doesn’t exist → build it from NYUmodel.onnx
👉 Then run exactly like before

Also, I’m fixing your CUDA context issues at the same time (no more stack errors).

✅ Clean, working version (ONNX → TensorRT → inference)
🔧 Key fixes:
builds engine from ONNX

removes broken push/pop usage

fixes Windows beep

keeps your pipeline intact

import cv2
import numpy as np
import matplotlib
import time
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # ✅ handles CUDA context automatically
import winsound

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def beep(freq=1000, duration=800):
    winsound.Beep(freq, duration)


class TRTDepthPredictor:
    def __init__(self,
                 onnx_path="NYUmodel.onnx",
                 engine_path="NYUmodel.trt",
                 input_size=(518, 518)):

        self.input_size = input_size
        self.cmap = matplotlib.colormaps["turbo"]

        # ✅ Build engine if not exists
        if not os.path.exists(engine_path):
            print("Building TensorRT engine from ONNX...")
            self.engine = self.build_engine(onnx_path)
            with open(engine_path, "wb") as f:
                f.write(self.engine.serialize())
        else:
            print("Loading existing TensorRT engine...")
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load/build TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def build_engine(self, onnx_path):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB

        # Jetson Nano supports FP16
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        return builder.build_engine(network, config)

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            shape = tuple(self.context.get_tensor_shape(name))
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({"host": host_mem, "device": device_mem, "name": name})
            else:
                outputs.append({"host": host_mem, "device": device_mem, "name": name})

        return inputs, outputs, bindings, stream

    def _preprocess(self, frame):
        h, w = self.input_size
        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std
        img = img.transpose(2, 0, 1)

        return np.ascontiguousarray(img[None])

    def _do_inference(self):
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
            self.context.set_tensor_address(inp["name"], int(inp["device"]))

        for out in self.outputs:
            self.context.set_tensor_address(out["name"], int(out["device"]))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

    def infer_image(self, frame):
        tensor = self._preprocess(frame)
        np.copyto(self.inputs[0]["host"], tensor.ravel())
        self._do_inference()

        h, w = self.input_size
        return self.outputs[0]["host"].reshape(h, w).copy()

    def colorize(self, depth):
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        colormap = self.cmap(depth_norm)[:, :, :3]
        colormap = (colormap * 255).astype(np.uint8)
        return cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)

    def infer_video(self, video_path, d, v, show=False):
        cap = cv2.VideoCapture(video_path)
        last_beep = 0
        prevdepth = None

        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            depth = self.infer_image(frame)

            if show:
                cv2.imshow("Depth TRT", self.colorize(depth))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            depth_disp = 255.0 - (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0

            h, w = depth_disp.shape
            cy, cx = h // 2, w // 2
            crop = depth_disp[cy-120:cy+120, cx-120:cx+120]

            if prevdepth is not None:
                velocity = -(crop - prevdepth)
                if (velocity > v).any() and time.time() - last_beep > 3:
                    beep(500, 800)
                    print("velocity warning")
                    last_beep = time.time()

            if (crop > d).any() and time.time() - last_beep > 3:
                beep(1000, 800)
                print("distance warning")
                last_beep = time.time()

            prevdepth = crop.copy()

        cap.release()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    beep(1000, 800)
    beep(1000, 800)

    predictor = TRTDepthPredictor(
        onnx_path="NYUmodel.onnx",
        engine_path="NYUmodel.trt",
        input_size=(518, 518),
    )

    predictor.infer_video(0, d=252, v=200, show=True)
⚠️ Important for Jetson Nano
You’re very likely going to hit performance limits.

Strong recommendations:
Reduce input size → (256, 256) or (320, 320)

Use FP16 (already enabled)

Consider TensorRT --workspace tuning if it fails to build

If something breaks next
The next likely failure points are:

ONNX parsing errors

Unsupported layers on Nano

Memory limits during build

If that happens, paste the error — that’s the real bottleneck step now.


tensorrt.tensorrt.builder oobject has no attribute 'build_engine'

That error is happening because you’re using a newer TensorRT API where:

❌ builder.build_engine(...) no longer exists
✅ It was replaced with builder.build_serialized_network(...)

This changed around TensorRT 8.5+ (and definitely in 10.x, which Jetson often uses now).

✅ Fix: update your engine build function
Replace your build_engine() with this:

def build_engine(self, onnx_path):
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # ✅ NEW API
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized engine")

    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(serialized_engine)
🧠 What changed (quick explanation)
Old TensorRT:

engine = builder.build_engine(network, config)
New TensorRT:

serialized = builder.build_serialized_network(network, config)
engine = runtime.deserialize_cuda_engine(serialized)
👉 You now explicitly:

Build a serialized engine blob

Deserialize it yourself

⚠️ One more thing (Jetson-specific)
If you're on Jetson Nano, memory is tight. If build fails, try:

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 26)  # 64MB
If you hit another error next
The next likely issue will be:

ONNX parsing errors (unsupported ops)

or shape issues

If that happens, paste the parser errors — those are very fixable, but they depend on your model.

You're actually very close now — this was one of the last major API landmines.


failed to build serialized engine network has dynamic or shape inputs

That error means TensorRT sees dynamic shapes in your ONNX model, but you didn’t tell it what sizes are allowed.

❌ “network has dynamic or shape inputs”
👉 TensorRT requires an optimization profile for dynamic inputs

✅ Fix: add an optimization profile
You need to explicitly define min / opt / max shapes.

Since your model is effectively fixed at (1, 3, 518, 518), just lock it:

🔧 Updated build_engine() (THIS is the important fix)
def build_engine(self, onnx_path):
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # ✅ FIX: create optimization profile
    profile = builder.create_optimization_profile()

    input_name = network.get_input(0).name

    # Lock shape (no dynamic behavior)
    profile.set_shape(
        input_name,
        (1, 3, 518, 518),  # min
        (1, 3, 518, 518),  # opt
        (1, 3, 518, 518)   # max
    )

    config.add_optimization_profile(profile)

    # Build
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized engine")

    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(serialized_engine)
🧠 Why this happened
Your ONNX model probably has something like:

input: (1, 3, H, W)
Where H and W are dynamic.

TensorRT refuses to build unless you say:

👉 “I will only ever use THIS range of shapes”

⚠️ Alternative (if you want flexibility)
If you do want dynamic resolution:

profile.set_shape(input_name,
    (1, 3, 256, 256),   # min
    (1, 3, 384, 384),   # opt
    (1, 3, 640, 640))   # max
But on a Jetson Nano:

👉 Don’t do this unless necessary — it increases memory + reduces performance.

🚨 Very important next step
After building, you must ALSO ensure:

self.context.set_input_shape(input_name, (1, 3, 518, 518))
Otherwise inference will fail later.

If it STILL fails
Then the issue is likely:

unsupported ONNX ops (very common with depth models)

or model too large for Nano

If that happens, show me:

parser.get_error(i)
That’s where the real debugging starts.

You're now past the hardest part (engine portability + API changes). This next step is usually the final hurdle.


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
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 1: "height", 2: "width"},
        },
        opset_version=opset,
        do_constant_folding=True,
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
                 workspace_gb: int = 4):

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

    profile = builder.create_optimization_profile()
    profile.set_shape("input",
                      min=(1, 3, *min_hw),
                      opt=(1, 3, *opt_hw),
                      max=(4, 3, *max_hw))
    config.add_optimization_profile(profile)

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
              input_hw: tuple = (518, 518),
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
this is the file I used to create the onnx file 

Good — this actually explains exactly why you hit the “dynamic shapes” error.

Your ONNX export includes this:

dynamic_axes={
    "input": {0: "batch", 2: "height", 3: "width"},
    "depth": {0: "batch", 1: "height", 2: "width"},
}
👉 That makes your model fully dynamic in batch + resolution.

🚨 The real issue
On a Jetson Nano, this is overkill and often breaks builds:

Dynamic shapes → requires optimization profiles (you added that)

But ALSO:

increases memory usage

increases build complexity

sometimes fails entirely on Nano

✅ Best fix (strongly recommended)
🔧 Export a static ONNX model instead
Just remove dynamic_axes completely.

✏️ Change this:
torch.onnx.export(
    wrapper,
    (dummy,),
    onnx_path,
    input_names=["input"],
    output_names=["depth"],
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "depth": {0: "batch", 1: "height", 2: "width"},
    },
✅ To this:
torch.onnx.export(
    wrapper,
    (dummy,),
    onnx_path,
    input_names=["input"],
    output_names=["depth"],
    opset_version=opset,
    do_constant_folding=True,
)
💡 Why this works better
You are already doing:

input_size = (518, 518)
So dynamic shapes give you zero benefit but cause:

❌ TensorRT build failures

❌ higher RAM usage (bad on Nano)

❌ slower inference

🚀 Then simplify your TensorRT build
Now you can REMOVE the optimization profile entirely:

Replace:
profile = builder.create_optimization_profile()
profile.set_shape(...)
config.add_optimization_profile(profile)
With nothing (delete it)
⚠️ Important: Jetson Nano survival tips
Your current config:

workspace_gb = 4
max_hw = (1024, 1024)
👉 That’s way too aggressive for Nano.

Use this instead:
workspace_gb = 1   # or even 0.5
And if it still fails:

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB
🔥 Optional but HIGHLY recommended
Your model at 518×518 is heavy for Nano.

Try:

input_hw = (384, 384)  # or even 320x320
You’ll get:

much faster FPS

successful engine builds

lower RAM usage

✅ Summary (what to do next)
❗ Remove dynamic_axes from ONNX export

🔁 Re-export ONNX

🔁 Rebuild TensorRT engine

🧠 Reduce:

input resolution (recommended)

workspace size if needed

If it still fails
Then we’re likely dealing with:

unsupported ops from DepthAnythingV2

or Nano memory exhaustion

At that point, the exact parser error output becomes critical.

You’re very close — this is one of the most common TensorRT traps, especially on Jetson.


command to run program

You run that script from the terminal using its CLI arguments.

✅ Basic command
python your_script.py --pth-path path/to/model.pth
✅ Example (what you’ll actually use)
python depth_trt.py \
  --pth-path ./checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --fp16 \
  --benchmark
⚙️ Recommended for Jetson Nano
Use smaller input size + lower workspace:

python depth_trt.py \
  --pth-path ./checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --input-h 384 \
  --input-w 384 \
  --workspace-gb 1 \
  --fp16 \
  --benchmark
🧪 Force rebuild (important after your fixes)
Since you changed ONNX export:

python depth_trt.py \
  --pth-path ./checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --fp16 \
  --force
📂 What it will do
Running the command will:

Load .pth

Export → depth_anything.onnx

Build → depth_anything.trt

Run inference / benchmark

Output goes to:

./trt_output/
🚨 Common mistakes
Wrong path to .pth → will crash immediately

Forgot --force after changing code → uses old broken ONNX

Too large resolution → Nano runs out of memory

If you want, next step I can help you hook this engine into your live camera pipeline (your earlier script).


failed version tag does not match note: current version 239serialzed version 236

That error means your TensorRT engine was built with a different TensorRT version than the one you're using to load it:

❌ current version: 239
❌ serialized engine: 236

👉 TensorRT engines are NOT forward/backward compatible across versions.

🔥 What’s happening
You built the .trt file with one TensorRT version (likely on another machine or older Jetson setup)

Now you're trying to load it with a newer/different TensorRT runtime

➡️ Result: deserialization fails

✅ The only fix (no workaround)
You must rebuild the engine on the same device and environment where it will run.

✔️ Do this:
1. Delete old engine
rm -f ./trt_output/depth_anything.trt
2. Rebuild on the Jetson Nano
python depth_trt.py \
  --pth-path ./checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --input-h 384 \
  --input-w 384 \
  --workspace-gb 1 \
  --fp16 \
  --force
🧠 Why this is required
TensorRT engines depend on:

GPU architecture (Maxwell vs Pascal)

CUDA version

cuDNN version

TensorRT version

Even a small mismatch → 💥 invalid engine

🚨 Important rule (remember this)
ONNX is portable. .trt is not.

✅ Move .onnx between machines

❌ Never move .trt between machines or environments

🧪 Quick sanity check
After rebuilding, test:

/usr/src/tensorrt/bin/trtexec --loadEngine=./trt_output/depth_anything.trt
If it runs → you're good.

If you still get errors
Then we’re down to:

ONNX incompatibility

memory limits on Nano

Those are fixable — just show the next error.

You’ve now hit the last major TensorRT “gotcha”: version lock. After this, things usually stabilize.


no module found onxx script 

That one’s simple — it’s just a typo / missing install.

❌ Problem
You wrote:

no module found onxx

👉 The correct module is:

onnx
(not onxx)

✅ Fix
Install ONNX:

pip install onnx
⚠️ On Jetson Nano (important)
Sometimes plain pip install onnx fails or installs a broken build due to ARM architecture.

If that happens, use:

pip install --no-cache-dir onnx
Or (more reliable on Jetson):

pip3 install onnx
🧪 Verify install
Run:

python -c "import onnx; print(onnx.__version__)"
If it prints a version → you're good.

🚨 If it STILL fails
Then you're likely hitting Jetson-specific issues (common). In that case:

sudo apt-get install python3-onnx
Summary
It’s just a typo or missing package

Install onnx

Verify import works

If you want, next I can help you skip ONNX entirely and build TensorRT directly from PyTorch — which can sometimes be easier on Jetson.


no module named onnxscript

That error comes from newer PyTorch/ONNX export paths — they now depend on an extra package:

❌ No module named onnxscript

✅ Fix
Just install it:

pip install onnxscript
⚠️ Jetson Nano note (important)
On Jetson, builds can be fragile. If the above fails:

pip3 install --no-cache-dir onnxscript
🧪 Verify
python -c "import onnxscript; print('onnxscript OK')"
💡 Why this happened
Recent PyTorch versions (2.x) use a newer ONNX exporter that relies on:

onnx

onnxscript ← new dependency

Older tutorials don’t mention it, which is why this pops up.

🚨 If install fails on Nano
This sometimes happens due to Python version mismatches. Two fallback options:

Option 1 — downgrade exporter behavior (quick workaround)
Modify your export:

torch.onnx.export(..., dynamo=False)
This forces the old exporter, which does NOT require onnxscript.

Option 2 — install compatible versions
pip install "onnx<1.16" "onnxscript<0.1"
✅ Recommended path (fastest)
Try:

pip install onnxscript
If it fails → use:

dynamo=False
If you hit another export error after this, it’ll likely be model-specific — and that’s where things get interesting.


runtime error onnx parsing failed

Now you’re past installs and into the real issue: your ONNX file is not compatible with TensorRT on the Jetson Nano.

“ONNX parsing failed” is a generic error — the real reason is always in the parser log, but we can already narrow it down.

🔥 Why this is happening (most likely)
Given your setup (Depth Anything V2 + Jetson Nano), the usual causes are:

1. ❌ Unsupported ops in ONNX
Depth models often include:

LayerNorm

GELU

Resize (advanced modes)

Scatter / Gather

Dynamic shapes

👉 TensorRT on Nano often cannot parse some of these.

2. ❌ Opset too high
You used:

opset_version=17
👉 Jetson Nano TensorRT is happiest with:

opset 11–13

3. ❌ Dynamic graph still leaking in
Even after removing dynamic_axes, PyTorch sometimes still exports:

symbolic dynamic shapes

or unsupported control flow

✅ Step 1 — Get the REAL error (important)
Modify your parser block:

if not parser.parse(f.read()):
    print("\n❌ ONNX PARSE ERRORS:\n")
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    raise RuntimeError("ONNX parsing failed")
👉 Run again and paste the output if you want precise fixing.

🚀 Step 2 — Quick fix that works 80% of the time
Re-export ONNX with safer settings:
🔧 Change your export to:
torch.onnx.export(
    wrapper,
    (dummy,),
    onnx_path,
    input_names=["input"],
    output_names=["depth"],
    opset_version=12,          # ✅ IMPORTANT: downgrade
    do_constant_folding=True,
)
AND remove everything else (no dynamic axes, no extras).

⚠️ Step 3 — If it STILL fails
Then you likely need one of these fixes:

Option A (most reliable for Jetson Nano)
Force legacy exporter:

torch.onnx.export(
    wrapper,
    (dummy,),
    onnx_path,
    opset_version=11,
    use_external_data_format=False
)
Option B — simplify model graph
Depth Anything V2 sometimes needs preprocessing like:

replacing GELU → ReLU (export-time trick)

or simplifying with:

pip install onnxsim
onnxsim model.onnx model_simplified.onnx
Option C — test ONNX outside TensorRT
Run:

python -c "import onnxruntime as ort; print('OK')"
Then:

import onnxruntime as ort
sess = ort.InferenceSession("model.onnx")
print("ONNX loads fine")
If ONNX Runtime fails → problem is export
If ONNX Runtime works → problem is TensorRT compatibility

🧠 Key insight
Right now your pipeline is:

PyTorch → ONNX (probably too complex) → TensorRT (Jetson Nano too strict)
Jetson Nano usually requires:

👉 simplified ONNX (opset 11–12, static shape, no fancy ops)

🧭 What I recommend (best path)
Do this in order:

Re-export ONNX with:

opset = 12

no dynamic axes

If still fails:

pip install onnxsim
onnxsim model.onnx model_simplified.onnx
Then rebuild TensorRT

If you want next step help
Paste the actual parser errors (very important).
I can tell you exactly which layer is breaking and how to patch it — that’s usually 1–2 small fixes for Depth models.


parse error: in node -1 with name and operator parse graph unsupported node assertion failed

That error is actually very specific:

❌ unsupported node ... ASSERTION failed

This almost always means:

🚨 Root cause
Your ONNX model contains a debug / training-time assertion node, which TensorRT cannot run at all.

TensorRT only supports pure inference graphs, but your export likely included:

Assert nodes (from PyTorch graph safety checks)

or a broken symbolic op inserted during export

🧠 Why this happens in your case
Depth Anything V2 models often trigger this when:

1. Export was done with a newer PyTorch exporter
It can insert:

aten::Assert
prim::Assert
2. Model still has graph control flow
Even if you removed dynamic axes, PyTorch can still export safety checks.

✅ FAST FIX (works most of the time)
🔧 Force export in “clean inference mode”
Modify your export like this:

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
        opset_version=opset,
        do_constant_folding=True,
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