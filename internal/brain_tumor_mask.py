import os
from pathlib import Path
import torch
import torch.nn as nn
import cv2
import numpy as np

# Define UNet Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(1, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.pool     = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(512, 1024)
        self.upconv4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.decoder4(torch.cat((self.upconv4(b), e4), dim=1))
        d3 = self.decoder3(torch.cat((self.upconv3(d4), e3), dim=1))
        d2 = self.decoder2(torch.cat((self.upconv2(d3), e2), dim=1))
        d1 = self.decoder1(torch.cat((self.upconv1(d2), e1), dim=1))
        return self.final_conv(d1)

# Load model and weights once at import
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Determine weights path relative to this file
data_dir = Path(__file__).parent / "storage"
weights_file = data_dir / "brain_tumor_segmentation.pth"
if not weights_file.exists():
    raise FileNotFoundError(f"Segmentation weights not found at {weights_file}")
# Initialize model
tumor_seg_model = UNet().to(device)
tumor_seg_model.load_state_dict(torch.load(str(weights_file), map_location=device))
tumor_seg_model.eval()

# Preprocessing for segmentation
def _preprocess_seg(image_path: str) -> torch.Tensor:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return img_tensor

# Public API: predict mask
def predict_mask(image_path: str) -> np.ndarray:
    """
    Generates a binary mask (0 or 255) for the tumor region.
    """
    img_tensor = _preprocess_seg(image_path)
    with torch.no_grad():
        output = tumor_seg_model(img_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
    # Threshold and convert to uint8 mask
    bin_mask = (mask > 0.5).astype(np.uint8) * 255
    return bin_mask
