
import cv2
import torch
import numpy as np
import matplotlib
import time
# removed winsound - linux alternative below
from depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingPredictor:
    def __init__(self, encoder="vitb", device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        model_configs = {
            "vits": {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            "vitb": {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            "vitl": {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        cfg = model_configs[encoder].copy()
        self.model = DepthAnythingV2(**cfg)
        ckpt = f"depth_anything_v2_{encoder}.pth"
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = self.model.to(self.device).eval()
        self.cmap = matplotlib.colormaps["turbo"]

    def beep(self, freq=1000, duration=0.8):
        # Linux beep alternative using speaker
        import os
        os.system(f"beep -f {freq} -l {int(duration * 1000)} 2>/dev/null || "
                  f"python3 -c \"import subprocess; subprocess.run(['paplay', '/usr/share/sounds/freedesktop/stereo/bell.oga'])\"")

    def colorize(self, depth):
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        colormap = self.cmap(depth_norm)[:, :, :3]
        colormap = (colormap * 255).astype(np.uint8)
        return cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)

    def infer_video(self, video_path, d, v, show=False):
        # Fix: use V4L2 backend for webcam on Linux
        if isinstance(video_path, int):
            cap = cv2.VideoCapture(video_path, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            # warm up camera
            time.sleep(2)
            for _ in range(30):
                cap.read()
        else:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        last_beep = 0
        prevdepth = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            depth = self.model.infer_image(frame)
            color = self.colorize(depth)

            if show:
                cv2.imshow("DepthAnythingV2", color)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            depth = 255.0 - (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            h, w = depth.shape
            cy, cx = h // 2, w // 2
            depth = depth[cy-120:cy+120, cx-120:cx+120]

            if prevdepth is not None:
                velocity = -(depth - prevdepth)
                if (velocity > v).any():
                    if time.time() - last_beep > 3:
                        self.beep(500, 0.8)
                        last_beep = time.time()
                        print("velocity warning")

            if (depth > d).any():
                if time.time() - last_beep > 3:
                    self.beep(1000, 0.8)
                    last_beep = time.time()
                    print("distance warning")

            prevdepth = depth.copy()

        cap.release()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    depth_model = DepthAnythingPredictor(encoder="vits")
    depth_model.infer_video(0, d=250, v=200, show=True)