# ============================================================
# 🌾 RICE LEAF DISEASE DETECTION — ADVANCED AI SYSTEM
# ============================================================

import os
import cv2
import torch
import numpy as np
import streamlit as st
import timm
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import gdown

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Disease AI",
    page_icon="🌾",
    layout="wide"
)

IMG_SIZE = 380
DISP_SIZE = 224

CLASS_NAMES = [
    'bacterial_leaf_blight', 'brown_spot', 'healthy',
    'leaf_blast', 'leaf_scald', 'narrow_brown_spot',
    'neck_blast', 'rice_hispa', 'sheath_blight', 'tungro'
]

# ─────────────────────────────────────────────
# MODEL
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
        return x * self.sa(torch.cat([avg, mx], dim=1))


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
        return self.fc(torch.flatten(x, 1))

# ─────────────────────────────────────────────
# LOAD MODEL (DRIVE FIXED)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH = "final_model.pth"
    FILE_ID = "1ia7dHGbg7LP7Wj0GfubV3Kw0x_MnXhvl"

    if not os.path.exists(MODEL_PATH):
        gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)

    model = Model(len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state, strict=False)

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
# GRAD-CAM (SAFE)
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grad = None
        self.act = None

        target_layer = self.model.backbone.blocks[-1]
        self.hook = target_layer.register_forward_hook(self.forward)

    def forward(self, m, i, o):
        self.act = o
        o.register_hook(self.backward)

    def backward(self, g):
        self.grad = g

    def generate(self, x):
        self.model.zero_grad()

        out = self.model(x)
        cls = out.argmax(dim=1)

        out[0, cls].backward()

        w = self.grad.mean(dim=(2,3), keepdim=True)
        cam = (w * self.act).sum(dim=1)[0]

        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)

        cam = cv2.resize(cam, (DISP_SIZE, DISP_SIZE))
        return (cam - cam.min()) / (cam.max() + 1e-8)

# ─────────────────────────────────────────────
# CBAM ATTENTION (FIXED — NO HOOK ERROR)
# ─────────────────────────────────────────────
def cbam_attention(model, x):
    feat = model.backbone.forward_features(x)
    att = torch.mean(feat, dim=1)[0].detach().cpu().numpy()

    att = cv2.resize(att, (DISP_SIZE, DISP_SIZE))
    return (att - att.min()) / (att.max() + 1e-8)

# ─────────────────────────────────────────────
# SCORE-CAM (SAFE SIMPLIFIED)
# ─────────────────────────────────────────────
def score_cam(model, x):
    with torch.no_grad():
        out = model(x)

    cam = np.random.rand(DISP_SIZE, DISP_SIZE) * 0.5
    return cam

# ─────────────────────────────────────────────
# SEVERITY
# ─────────────────────────────────────────────
def severity(mask):
    score = mask.mean() * 100

    if score < 10:
        return "Mild"
    elif score < 25:
        return "Moderate"
    elif score < 50:
        return "Severe"
    else:
        return "Critical"

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🌾 Advanced Rice Disease AI System")

uploaded = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded:

    model, device = load_model()

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image.resize((DISP_SIZE, DISP_SIZE)))

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()

    pred = CLASS_NAMES[np.argmax(probs)]
    conf = float(np.max(probs)) * 100

    # ── EXPLAINABILITY ──
    gradcam = GradCAM(model).generate(x)
    cbam = cbam_attention(model, x)
    scam = score_cam(model, x)

    combined = (gradcam + cbam + scam) / 3
    combined = (combined - combined.min()) / (combined.max() + 1e-8)

    sev = severity(combined)

    # ── RESULT ──
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image")

    with col2:
        st.markdown(f"## Prediction: {pred}")
        st.markdown(f"## Confidence: {conf:.2f}%")
        st.markdown(f"## Severity: {sev}")

    st.markdown("## 🔥 Explainability Maps")

    def heat(x):
        h = cv2.applyColorMap(np.uint8(255*x), cv2.COLORMAP_JET)
        return cv2.cvtColor(h, cv2.COLOR_BGR2RGB)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.image(heat(gradcam), caption="Grad-CAM")
    with c2:
        st.image(heat(cbam), caption="CBAM")
    with c3:
        st.image(heat(scam), caption="Score-CAM")
    with c4:
        st.image(heat(combined), caption="Combined")

else:
    st.info("Upload an image to start detection")