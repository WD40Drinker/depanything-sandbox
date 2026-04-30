import cv2
import numpy as np
import matplotlib
import time
import os
import onnxruntime as ort

def beep(freq=1000, duration=800):
    os.system(f'beep -f {freq} -l {duration}')


# =========================================================
# DEPTH ANYTHING (ONNX ONLY)
# =========================================================
class ONNXDepthPredictor:
    def __init__(self,
                 onnx_path="depth_anything.onnx",
                 input_size=(224, 224)):

        self.input_size = input_size
        self.cmap = matplotlib.colormaps["turbo"]

        print("[INFO] Loading ONNX model...")

        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]  # safest for Jetson Nano
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # =====================================================
    # PREPROCESS
    # =====================================================
    def preprocess(self, frame):
        h, w = self.input_size

        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std
        img = img.transpose(2, 0, 1)

        return np.expand_dims(img, 0).astype(np.float32)

    # =====================================================
    # INFERENCE
    # =====================================================
    def infer(self, frame):
        inp = self.preprocess(frame)

        depth = self.session.run(
            [self.output_name],
            {self.input_name: inp}
        )[0]

        return depth[0]

    # =====================================================
    # COLORIZE
    # =====================================================
    def colorize(self, depth):
        d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        col = self.cmap(d)[:, :, :3]
        return (col * 255).astype(np.uint8)

    # =====================================================
    # 🔥 SAME VIDEO FUNCTION YOU WANTED
    # =====================================================
    def infer_video(self, video_source, d, v, show=False):
        cap = cv2.VideoCapture(video_source)

        last_beep = 0
        prevdepth = None

        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            depth = self.infer(frame)

            # ---------------- display ----------------
            if show:
                cv2.imshow("Depth ONNX", self.colorize(depth))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ---------------- processing ----------------
            depth_disp = 255.0 - (
                (depth - depth.min()) /
                (depth.max() - depth.min() + 1e-8)
            ) * 255.0

            h, w = depth_disp.shape
            cy, cx = h // 2, w // 2

            crop = depth_disp[cy-120:cy+120, cx-120:cx+120]

            # ---------------- velocity check ----------------
            if prevdepth is not None:
                velocity = -(crop - prevdepth)

                if (velocity > v).any() and time.time() - last_beep > 3:
                    beep(500, 800)
                    print("velocity warning")
                    last_beep = time.time()

            # ---------------- distance check ----------------
            if (crop > d).any() and time.time() - last_beep > 3:
                beep(1000, 800)
                print("distance warning")
                last_beep = time.time()

            prevdepth = crop.copy()

        cap.release()
        if show:
            cv2.destroyAllWindows()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    beep(1000, 800)

    predictor = ONNXDepthPredictor(
        onnx_path="depth_anything.onnx",
        input_size=(224, 224)
    )

    predictor.infer_video(
        video_source=0,
        d=252,
        v=200,
        show=True
    )