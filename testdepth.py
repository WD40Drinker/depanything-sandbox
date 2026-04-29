import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import torch

# Load model
cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
model = DepthAnythingV2(**cfg)
model.load_state_dict(torch.load("depth_anything_v2_vits.pth", map_location="cpu"))
model = model.eval()

# Test with dummy image
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
depth = model.infer_image(dummy)
print("Depth shape:", depth.shape)