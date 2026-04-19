# ============================================================
# RICE LEAF DISEASE DETECTION — STREAMLIT APP (FINAL DEPLOYMENT)
# ============================================================

import os
import cv2
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
import gdown

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Disease Detector",
    page_icon="🌾",
    layout="wide"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE = 380
DISP_SIZE = 224

CLASS_NAMES = [
    'bacterial_leaf_blight', 'brown_spot', 'healthy',
    'leaf_blast', 'leaf_scald', 'narrow_brown_spot',
    'neck_blast', 'rice_hispa', 'sheath_blight', 'tungro'
]

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 16, 1), nn.ReLU(),
            nn.Conv2d(c // 16, c, 1), nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sa(torch.cat([avg, mx], dim=1))
        return x


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=False,
            num_classes=0,
            global_pool=''
        )
        c = self.backbone.num_features

        self.cbam = CBAM(c)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(c, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ─────────────────────────────────────────────
# LOAD MODEL (GOOGLE DRIVE FIXED)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH = "final_model.pth"
    FILE_ID = "1ia7dHGbg7LP7Wj0GfubV3Kw0x_MnXhvl"

    # download model if not exists
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)

    model = Model(len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()

    return model, device

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# GRAD-CAM (FIXED LAYER)
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grad = None
        self.act = None

        target_layer = self.model.backbone.features[-1]
        self.hook = target_layer.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, inp, out):
        self.act = out
        out.register_hook(self.backward_hook)

    def backward_hook(self, grad):
        self.grad = grad

    def generate(self, x):
        self.model.zero_grad()

        out = self.model(x)
        class_idx = out.argmax(dim=1)

        score = out[0, class_idx]
        score.backward()

        weights = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.act).sum(dim=1)

        cam = cam[0].detach().cpu().numpy()
        cam = np.maximum(cam, 0)

        cam = cv2.resize(cam, (DISP_SIZE, DISP_SIZE))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

    def remove(self):
        self.hook.remove()

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🌾 Rice Leaf Disease Detection (AI System)")

uploaded = st.file_uploader("Upload Rice Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded:

    model, device = load_model()

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image.resize((DISP_SIZE, DISP_SIZE)))

    x = transform(image).unsqueeze(0).to(device)

    with st.spinner("Analyzing leaf..."):
        with torch.no_grad():
            out = model(x)
            probs = F.softmax(out, dim=1)[0].cpu().numpy()

        pred = CLASS_NAMES[np.argmax(probs)]
        conf = float(np.max(probs)) * 100

    # ── Result
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image")

    with col2:
        st.markdown(f"## Prediction: {pred}")
        st.markdown(f"## Confidence: {conf:.2f}%")

    # ── Grad-CAM
    if pred != "healthy":
        st.markdown("### 🔥 Disease Attention Map")

        cam_engine = GradCAM(model)
        cam = cam_engine.generate(x)
        cam_engine.remove()

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        st.image(overlay, caption="Grad-CAM Visualization")

else:
    st.info("Upload a rice leaf image to start prediction")