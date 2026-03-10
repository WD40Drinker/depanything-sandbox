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
from depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingPredictor:
    def __init__(self, encoder="vitb", device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        model_configs = {
            "vits": {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            "vitb": {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            "vitl": {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        if encoder not in model_configs:
            raise ValueError(f"Invalid encoder: {encoder}")

        # Load model
        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f"depth_anything_v2_{encoder}.pth", map_location="cpu"))
        self.model = self.model.to(self.device).eval()

        # Default colormap
        self.cmap = matplotlib.colormaps["turbo"]

    def infer_image(self, image):
        """Run depth estimation on a single image."""
        depth = self.model.infer_image(image)  # float32 depth tensor
        return depth

    def colorize(self, depth):
        """Convert depth map to turbo colormap."""
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        colormap = self.cmap(depth_norm)[:, :, :3]  # RGB float (0-1)
        colormap = (colormap * 255).astype(np.uint8)
        return cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)

    def infer_and_save_image(self, img_path, save_path="depth.png"):
        """Infer depth from an image and save colormap."""
        img = cv2.imread(img_path)
        depth = self.infer_image(img)
        color = self.colorize(depth)
        cv2.imwrite(save_path, color)
        return depth, color

    def infer_video(self, video_path, save_path="depth_video.mp4"):
        """Run depth estimation on video and save colored depth video."""
        cap = cv2.VideoCapture(video_path)

        # Output video settings
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            depth = self.infer_image(frame)
            color = self.colorize(depth)
            writer.write(color)

            # Optional live preview
            cv2.imshow("DepthAnythingV2", color)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        writer.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    depth_model = DepthAnythingPredictor(encoder="vits")

    # Inference on image
    # depth, color = depth_model.infer_and_save_image("bus.jpg", save_path="depth_turbo.png")

    # Inference on video
    #depth_model.infer_video("fish.mp4", "depth_output.mp4")

    # Inference on webcam
    #depth_model.infer_video(0, "depth_webcam.mp4")

    #Save webcam feed to depth variable
    raw_video = cv2.VideoCapture(0)
    #output_width = frame_width
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        
        depth = depth_model.infer_image(raw_frame)
        #print(depth)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        print(depth)
        velocity = prevdepth - depth 
        prevdepth = depth

    raw_video.release()
    #out.release()