import cv2
import torch
import numpy as np
import matplotlib
import time
import torch.nn.functional as F
from depth_anything_v2.dpt import DepthAnythingV2


def infer(model, frame, device, input_size=518):
    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize
    h, w = img.shape[:2]
    scale = input_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    # make divisible by 14
    new_h = (new_h // 14) * 14
    new_w = (new_w // 14) * 14
    img = cv2.resize(img, (new_w, new_h))

    # Normalize and move to device
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = (img / 255.0 - mean) / std

    tensor = torch.from_numpy(img).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # to GPU

    with torch.no_grad():
        depth = model(tensor)

    # Resize back to original
    depth = F.interpolate(
        depth.unsqueeze(1),
        size=(h, w),
        mode="bilinear",
        align_corners=True
    ).squeeze().cpu().numpy()

    return depth


class DepthAnythingPredictor:
    def __init__(self, encoder="vits", device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        model_configs = {
            "vits": {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            "vitb": {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            "vitl": {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        cfg = model_configs[encoder].copy()
        self.model = DepthAnythingV2(**cfg)
        self.model.load_state_dict(torch.load(
            f"depth_anything_v2_{encoder}.pth",
            map_location="cpu",
            weights_only=False
        ))
        self.model = self.model.to(self.device).eval()
        self.cmap = matplotlib.colormaps["turbo"]

    def beep(self, freq=1000, duration=0.8):
        import os
        os.system("paplay /usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null &")

    def colorize(self, depth):
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        colormap = self.cmap(depth_norm)[:, :, :3]
        colormap = (colormap * 255).astype(np.uint8)
        return cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)

    def infer_video(self, video_path, d, v, show=False):
        if isinstance(video_path, int):
            cap = cv2.VideoCapture(video_path, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
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

            # use manual inference instead of infer_image
            depth = infer(self.model, frame, self.device)

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