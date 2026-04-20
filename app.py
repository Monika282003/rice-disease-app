# ============================================================
# 🌾 RICE LEAF DISEASE DETECTION — FINAL DEPLOYABLE VERSION
# EfficientNet-B4 + CBAM + GAP + MLP
# ============================================================

import os, cv2, torch, requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Rice Disease AI", layout="wide")
st.title("🌾 Rice Leaf Disease Detection")

IMG_SIZE = 380
DISP_SIZE = 224

CLASS_NAMES = [
    'bacterial_leaf_blight','brown_spot','healthy',
    'leaf_blast','leaf_scald','narrow_brown_spot',
    'neck_blast','rice_hispa','sheath_blight','tungro'
]

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ─────────────────────────────────────────────
# CBAM
# ─────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//16, 1),
            nn.ReLU(),
            nn.Conv2d(channels//16, channels, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2,1,7,padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx,_ = torch.max(x, dim=1, keepdim=True)
        return x * self.sa(torch.cat([avg,mx], dim=1))

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=False,
            num_classes=0,
            global_pool=''
        )

        ch = self.backbone.num_features

        self.cbam = CBAM(ch)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(ch,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        return self.fc(x)

# ─────────────────────────────────────────────
# LOAD MODEL FROM GOOGLE DRIVE
# ─────────────────────────────────────────────
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1ia7dHGbg7LP7Wj0GfubV3Kw0x_MnXhvl"

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(len(CLASS_NAMES))

    model_path = "final_model.pth"

    # ✅ Proper Google Drive download
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading model..."):
            gdown.download(MODEL_URL, model_path, quiet=False)

    # ✅ Load safely
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
    except Exception as e:
        st.error("❌ Model loading failed. File may be corrupted.")
        st.stop()

    model.to(device)
    model.eval()

    return model, device

# ✅ IMPORTANT
model, device = load_model()

# ─────────────────────────────────────────────
# GRADCAM++
# ─────────────────────────────────────────────
class GradCAMPlusPlus:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None

        layer.register_forward_hook(self.forward_hook)
        layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, m, i, o):
        self.activations = o

    def backward_hook(self, m, gi, go):
        self.gradients = go[0]

    def generate(self, x):
        self.model.zero_grad()
        out = self.model(x)
        class_idx = out.argmax()

        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1),
                            size=(DISP_SIZE, DISP_SIZE),
                            mode='bilinear').squeeze()

        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max()+1e-8)
        return cam

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
uploaded = st.file_uploader("📤 Upload Rice Leaf Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Image")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]

    pred = CLASS_NAMES[np.argmax(probs)]
    conf = np.max(probs) * 100

    st.success(f"Prediction: {pred.replace('_',' ').title()}")
    st.info(f"Confidence: {conf:.2f}%")

    # Grad-CAM
    cam_extractor = GradCAMPlusPlus(model, model.backbone.blocks[-1])
    cam = cam_extractor.generate(x)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_np = np.array(image.resize((DISP_SIZE, DISP_SIZE)))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Grad-CAM Visualization")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("EfficientNet-B4 + CBAM + GAP + MLP | Grad-CAM++")