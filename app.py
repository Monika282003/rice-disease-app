# ============================================================
# RICE LEAF DISEASE DETECTION — FINAL STABLE STREAMLIT APP
# ============================================================

import os, cv2, torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from io import BytesIO

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(page_title="Rice Disease AI", page_icon="🌾", layout="wide")

# ── Constants ───────────────────────────────────────────────
IMG_SIZE = 380
DISP_SIZE = 224

CLASS_NAMES = [
    'bacterial_leaf_blight', 'brown_spot', 'healthy',
    'leaf_blast', 'leaf_scald', 'narrow_brown_spot',
    'neck_blast', 'rice_hispa', 'sheath_blight', 'tungro'
]

# ── Model ───────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=False,
            num_classes=0,
            global_pool=''
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ── Load Model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(len(CLASS_NAMES))

    state = torch.load("final_model.pth", map_location=device)
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model, device

# ── Transform ───────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ── Grad-CAM (SAFE VERSION) ────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grad = None
        self.act = None

        self.hook = model.backbone.blocks[-1].register_forward_hook(self.forward_hook)

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

        weights = self.grad.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.act).sum(dim=1)

        cam = cam[0].detach().cpu().numpy()
        cam = np.maximum(cam, 0)

        cam = cv2.resize(cam, (DISP_SIZE, DISP_SIZE))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

    def remove(self):
        self.hook.remove()

# ── Score-CAM (STABLE VERSION) ─────────────────────────────


 def score_cam(model, x, class_idx, device):
    model.eval()
    activations = []

    handle = model.backbone.blocks[-1].register_forward_hook(
        lambda m,i,o: activations.append(o)
    )

    with torch.no_grad():
        base = F.softmax(model(x), dim=1)[0, class_idx].item()

    handle.remove()

    acts = activations[0][0].detach().cpu()

    cam = torch.zeros((DISP_SIZE, DISP_SIZE))

    for i in range(min(10, acts.shape[0])):

        a = acts[i]

        a = F.interpolate(
            a.unsqueeze(0).unsqueeze(0),
            size=(DISP_SIZE, DISP_SIZE),
            mode='bilinear'
        ).squeeze()

        a = (a - a.min()) / (a.max() + 1e-8)

        # 🔥 FIX HERE (CRITICAL)
        a_3ch = a.unsqueeze(0).repeat(3, 1, 1)   # [3,H,W]

        x_mask = x.clone()
        x_mask = x_mask * a_3ch.to(device)

        with torch.no_grad():
            score = F.softmax(model(x_mask), dim=1)[0, class_idx].item()

        cam += score * a.cpu()

    cam = cam.numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    return cam   

# ── Helpers ────────────────────────────────────────────────
def to_heatmap(x):
    h = cv2.applyColorMap(np.uint8(255*x), cv2.COLORMAP_JET)
    return cv2.cvtColor(h, cv2.COLOR_BGR2RGB)

# ── UI ─────────────────────────────────────────────────────
st.title("🌾 Rice Disease Detection AI (Fixed Version)")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    model, device = load_model()

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image.resize((DISP_SIZE, DISP_SIZE)))

    x = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()

    pred = CLASS_NAMES[np.argmax(probs)]
    conf = np.max(probs)*100

    col1, col2 = st.columns(2)

    with col1:
        st.image(image)

    with col2:
        st.markdown(f"## Prediction: {pred}")
        st.markdown(f"## Confidence: {conf:.2f}%")

    # ── Explainability ─────────────────────────────
    if pred != "healthy":

        st.markdown("## 🔥 Explainability Maps")

        gradcam = GradCAM(model)
        cam1 = gradcam.generate(x)
        gradcam.remove()

        cam2 = score_cam(model, x, int(np.argmax(probs)), device)

        col1, col2 = st.columns(2)

        with col1:
            st.image(to_heatmap(cam1), caption="Grad-CAM++")
        with col2:
            st.image(to_heatmap(cam2), caption="Score-CAM")

        combined = (cam1 + cam2)/2
        st.image(to_heatmap(combined), caption="Combined Attention")

else:
    st.info("Upload image to start detection")