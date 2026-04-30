import cv2
import numpy as np
import matplotlib
import winsound
import time

import tensorrt as trt
import pycuda.driver as cuda

# ── Manual CUDA context (replaces pycuda.autoinit) ───────────────────────────
cuda.init()
cuda_ctx = cuda.Device(0).make_context()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


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
                        winsound.Beep(500, 800)
                        last_beep = time.time()
                        print("velocity warning")

            if (crop > d).any():
                if time.time() - last_beep > 3:
                    winsound.Beep(1000, 800)
                    last_beep = time.time()
                    print("distance warning")

            prevdepth = crop.copy()

        cap.release()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        winsound.Beep(1000, 200)
        winsound.Beep(1000, 200)

        predictor = TRTDepthPredictor(
            engine_path = "NYUmodel.trt",
            input_size  = (518, 518),
        )
        predictor.infer_video(0, d=252, v=200, show=True)

    finally:
        cuda_ctx.pop()  # always release context on exit