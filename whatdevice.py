import torch
from depth_anything_v2.dpt import DepthAnythingV2

cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
model = DepthAnythingV2(**cfg)
model.load_state_dict(torch.load(
    "depth_anything_v2_vits.pth",
    map_location="cpu",
    weights_only=False
))
model = model.to("cuda").eval()

# Check all params are on cuda
print(next(model.parameters()).device)

# Test with dummy frame
import numpy as np
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
with torch.no_grad():
    depth = model.infer_image(dummy)
print("depth shape:", depth.shape)