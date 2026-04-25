# ==========================================================================
# Step-01: git clone https://github.com/DepthAnything/Depth-Anything-V2

# Step-02: cd Depth-Anything-V2

# Step-03: pip install -r requirements.txt

# Step-04: Download Depth-Anything-V2-Base model checkpoint from the link, and add inside the directory:
# https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true
# https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vits.pth?download=true


# I also did this but idk if you need to: pip install depth_anything_v2

# After this, you can use the code mentioned below, make sure this file is inside Depth-Anything-V2 directory.
# ===========================================================================

import cv2
import torch
import numpy as np
import matplotlib
import winsound
import time
from depth_anything_v2.dpt import DepthAnythingV2
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as MetricDepthAnything


class DepthAnythingPredictor:
    def __init__(self, encoder="vitb", device=None, metric=False, dataset="hypersim"):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.metric = metric

        model_configs = {
            "vits": {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            "vitb": {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            "vitl": {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        if encoder not in model_configs:
            raise ValueError(f"Invalid encoder: {encoder}")

        cfg = model_configs[encoder].copy()

        if metric:
            max_depth = 20 if dataset == "hypersim" else 80
            self.model = MetricDepthAnything(**{**cfg, 'max_depth': max_depth})
            ckpt = "model.pth"
        else:
            self.model = DepthAnythingV2(**cfg)
            ckpt = f"depth_anything_v2_{encoder}.pth"

        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = self.model.to(self.device).eval()
        self.cmap = matplotlib.colormaps["turbo"]

    def infer_image(self, image):
        return self.model.infer_image(image)

    def colorize(self, depth):
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        colormap = self.cmap(depth_norm)[:, :, :3]
        colormap = (colormap * 255).astype(np.uint8)
        return cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)

    def infer_video(self, video_path, show=False):
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

            color = self.colorize(depth)
            if show:
                cv2.imshow("DepthAnythingV2", color)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            depth = 255.0 - (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            # Center crop 240x240
            h, w = depth.shape
            cy, cx = h // 2, w // 2
            depth = depth[cy-120:cy+120, cx-120:cx+120]

            if prevdepth is not None:
                velocity = depth - prevdepth
                print(velocity)
                
                if (velocity < -60).any():
                    if time.time() - last_beep > 3:
                        winsound.Beep(500, 800)
                        last_beep = time.time()

            
            if (depth > 250).any():
                if time.time() - last_beep > 3:
                    winsound.Beep(1000, 800)
                    last_beep = time.time()
            

            prevdepth = depth.copy()

        cap.release()
        if show:
            cv2.destroyAllWindows()



if __name__ == "__main__":

    depth_model = DepthAnythingPredictor(encoder="vits", metric=True, dataset="hypersim")
    ckpt = torch.load("model.pth", map_location="cpu")
    print(list(ckpt.keys())[:10])  # first 10 keys

    winsound.Beep(1000, 200)
    winsound.Beep(1000, 200)

    # show=False avoids the cv2.imshow crash if you don't have GUI OpenCV
    depth_model.infer_video(0, show=True)

