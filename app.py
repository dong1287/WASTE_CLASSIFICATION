# app.py — Recyclable vs Organic 분류 웹 서비스
import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown

# ── 페이지 설정 ───────────────────────────────────────────
st.set_page_config(page_title="Recyclable vs Organic 분류기", page_icon="♻️")

# ── 모델 설정 ────────────────────────────────────────────
MODEL_URL = "https://drive.google.com/uc?id=1LYbh5Br1j5rAAiMLwbiSs-atzVysNeue"
MODEL_PATH = "best_model.pt"

# ── 모델 로드 (앱 시작 시 1회만 실행) ─────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    )
    model.eval()
    return model

model = load_model()

# ── 이미지 전처리 (학습 시 eval_transform과 동일) ──────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── 앱 제목/설명 ──────────────────────────────────────────
st.title("♻️ Recyclable vs Organic 분류기")
st.caption("이미지를 업로드하면 재활용 가능 폐기물인지, 유기물인지 분류합니다.")

# ── 이미지 업로드 ─────────────────────────────────────────
uploaded = st.file_uploader(
    "이미지를 선택하세요", type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_container_width=True)

    # ── 예측 ──────────────────────────────────────────────
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logit = model(input_tensor)
        prob = torch.sigmoid(logit).item()

    # class_to_idx 가 {'organic': 0, 'recyclable': 1} 라고 가정
    is_recyclable = prob >= 0.5
    label = "♻️ Recyclable" if is_recyclable else "🍃 Organic"
    confidence = prob if is_recyclable else 1 - prob

    # ── 결과 표시 ─────────────────────────────────────────
    st.markdown(f"### 예측 결과: {label}")
    st.metric("확신도", f"{confidence:.1%}")

    # 확률 바 시각화
    col1, col2 = st.columns(2)
    with col1:
        st.write("🍃 Organic")
        st.progress(float(1 - prob))
    with col2:
        st.write("♻️ Recyclable")
        st.progress(float(prob))

else:
    st.info("이미지를 업로드하면 분류 결과를 확인할 수 있습니다.")