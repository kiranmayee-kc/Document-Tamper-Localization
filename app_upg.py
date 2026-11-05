import os
import sys

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as TF


# ============================================================
# PROJECT CONFIGURATION — UNCHANGED
# ============================================================

sys.path.insert(0, os.path.abspath("."))

from src.demo_inference import load_model


CHECKPOINT = "checkpoints/best_model.pt"
TARGET_SIZE = (512, 512)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
THRESHOLD = 0.5


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TamperDoc - Document Tamper Localization",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# FRONTEND CSS ONLY
# ============================================================

st.markdown(
    """
    <style>
        /* ---------- Global ---------- */
        .stApp {
            background: #f7f8fc;
        }

        .main .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        /* ---------- Header ---------- */
        .hero {
            background: linear-gradient(135deg, #172554 0%, #1e3a8a 55%, #2563eb 100%);
            border-radius: 20px;
            padding: 30px 34px;
            margin-bottom: 24px;
            box-shadow: 0 12px 30px rgba(30, 58, 138, 0.16);
        }

        .hero-title {
            color: white;
            font-size: 2.15rem;
            font-weight: 750;
            letter-spacing: -0.6px;
            margin: 0;
        }

        .hero-subtitle {
            color: #dbeafe;
            font-size: 1rem;
            margin-top: 8px;
            margin-bottom: 0;
            line-height: 1.55;
        }

        .hero-badge {
            display: inline-block;
            margin-top: 17px;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.20);
            color: #eff6ff;
            font-size: 0.82rem;
            font-weight: 650;
        }

        /* ---------- Upload ---------- */
        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #172033;
            margin: 0 0 8px 0;
        }

        .section-subtitle {
            color: #667085;
            font-size: 0.92rem;
            margin-bottom: 14px;
        }

        .upload-card {
            background: white;
            border: 1px solid #e4e7ec;
            border-radius: 16px;
            padding: 20px 22px 10px 22px;
            box-shadow: 0 5px 18px rgba(16, 24, 40, 0.05);
            margin-bottom: 22px;
        }

        /* ---------- Status / result ---------- */
        .result-card {
            background: white;
            border: 1px solid #e4e7ec;
            border-radius: 18px;
            padding: 22px;
            margin: 18px 0 24px 0;
            box-shadow: 0 7px 24px rgba(16, 24, 40, 0.06);
        }

        .result-label {
            color: #667085;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .result-status {
            font-size: 1.7rem;
            font-weight: 780;
            margin: 0 0 4px 0;
        }

        .result-status.detected {
            color: #b42318;
        }

        .result-status.clear {
            color: #027a48;
        }

        .result-description {
            color: #667085;
            font-size: 0.9rem;
            margin: 0;
        }

        .metric-card {
            background: #f8fafc;
            border: 1px solid #eaecf0;
            border-radius: 13px;
            padding: 14px 15px;
            min-height: 84px;
        }

        .metric-label {
            color: #667085;
            font-size: 0.76rem;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .metric-value {
            color: #101828;
            font-size: 1.28rem;
            font-weight: 750;
        }

        /* ---------- Output cards ---------- */
        .output-card {
            background: white;
            border: 1px solid #e4e7ec;
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 22px;
            box-shadow: 0 7px 24px rgba(16, 24, 40, 0.05);
        }

        .output-title {
            color: #172033;
            font-size: 1.18rem;
            font-weight: 720;
            margin-bottom: 3px;
        }

        .output-description {
            color: #667085;
            font-size: 0.84rem;
            margin-bottom: 12px;
        }

        .map-note {
            display: inline-block;
            background: #f2f4f7;
            color: #475467;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.76rem;
            font-weight: 600;
            margin-bottom: 8px;
        }

        /* ---------- Information panels ---------- */
        .info-card {
            background: white;
            border: 1px solid #e4e7ec;
            border-radius: 16px;
            padding: 20px;
            margin-top: 8px;
            box-shadow: 0 5px 18px rgba(16, 24, 40, 0.04);
        }

        .info-title {
            color: #172033;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            padding: 9px 0;
            border-bottom: 1px solid #f2f4f7;
            font-size: 0.88rem;
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-key {
            color: #667085;
        }

        .info-value {
            color: #101828;
            font-weight: 650;
            text-align: right;
        }

        /* ---------- Streamlit controls ---------- */
        div[data-testid="stFileUploader"] {
            background: #f8fafc;
            border: 1.5px dashed #98a2b3;
            border-radius: 14px;
            padding: 8px;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 11px;
            min-height: 48px;
            font-weight: 700;
            font-size: 0.98rem;
            border: none;
            box-shadow: 0 5px 14px rgba(37, 99, 235, 0.18);
        }

        div[data-testid="stImage"] img {
            border-radius: 12px;
        }

        /* Reduce default Streamlit vertical gaps */
        div[data-testid="stVerticalBlock"] > div {
            gap: 0.65rem;
        }

        /* ---------- Footer ---------- */
        .footer {
            text-align: center;
            color: #98a2b3;
            font-size: 0.78rem;
            padding-top: 28px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Document Tamper Localization</div>
        <div class="hero-subtitle">
            Analyze document images using the trained TamperDoc model and obtain
            pixel-level tamper probability and localization results.
        </div>
        <div class="hero-badge">ResNet + DCT + SRM &nbsp;•&nbsp; 512 × 512 Input</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL — UNCHANGED
# ============================================================

@st.cache_resource
def get_model():
    model, checkpoint = load_model(CHECKPOINT)
    model.eval()
    return model, checkpoint


# ============================================================
# UPLOAD SECTION — FRONTEND ONLY
# ============================================================

st.markdown('<div class="section-title">Upload Document</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Upload a JPG, JPEG, or PNG document image. The model requires an exact resolution of 512 × 512 pixels.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Choose a document image",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )


if uploaded_file is None:
    st.markdown(
        '<div class="footer">Upload a document to begin tamper localization.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# READ IMAGE — UNCHANGED
# ============================================================

try:
    pil_img = Image.open(uploaded_file).convert("RGB")
except Exception as e:
    st.error(f"Unable to open image: {e}")
    st.stop()


# ============================================================
# IMAGE DIMENSION CHECK — UNCHANGED
# ============================================================

if pil_img.size != TARGET_SIZE:
    st.error(
        f"Invalid image size: {pil_img.size}. "
        f"The model requires exactly {TARGET_SIZE[0]}×{TARGET_SIZE[1]} pixels."
    )
    st.stop()


# ============================================================
# UPLOAD INFO + RUN BUTTON
# ============================================================

info_col, action_col = st.columns([2.2, 1], gap="large")

with info_col:
    st.markdown(
        f"""
        <div class="info-card" style="margin-top:0;">
            <div class="info-title">Document Ready</div>
            <div class="info-row">
                <span class="info-key">File</span>
                <span class="info-value">{uploaded_file.name}</span>
            </div>
            <div class="info-row">
                <span class="info-key">Resolution</span>
                <span class="info-value">512 × 512 pixels</span>
            </div>
            <div class="info-row">
                <span class="info-key">Input Channels</span>
                <span class="info-value">RGB · 3 channels</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with action_col:
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    run_detection = st.button("🔍  Run Tamper Detection", type="primary")


if not run_detection:
    st.markdown(
        '<div class="footer">Click <b>Run Tamper Detection</b> to analyze the uploaded document.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# ACTUAL MODEL INFERENCE — DO NOT CHANGE
# ============================================================

with st.spinner("Running TamperDoc model..."):

    # 1. Convert image to tensor
    raw_x = TF.to_tensor(pil_img).unsqueeze(0)

    # 2. ImageNet normalization for RGB branch
    norm_x = TF.normalize(
        raw_x.clone(),
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD
    )

    # 3. Load trained model
    try:
        model, checkpoint = get_model()
    except Exception as e:
        st.error(f"Unable to load model checkpoint: {e}")
        st.stop()

    # 4. Actual model inference
    with torch.inference_mode():
        outputs = model(
            norm_x,
            raw_x=raw_x
        )

        # Raw model output
        logits = outputs["tamper_logits"]

        # Convert logits to probabilities
        probability = torch.sigmoid(logits)

        # Binary prediction
        predicted_mask = (
            probability >= THRESHOLD
        )

    # 5. Convert outputs to NumPy
    probability_map = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )

    predicted_mask = (
        predicted_mask
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    logits_map = (
        logits
        .squeeze()
        .cpu()
        .numpy()
    )

    # 6. Sanity check
    if (
        np.isnan(logits_map).any()
        or np.isinf(logits_map).any()
    ):
        st.error("Model output contains NaN or Inf values.")
        st.stop()

    # COMPUTE STATISTICS — SAME VALUES AS EXISTING APP
    prob_min = float(np.min(probability_map))
    prob_max = float(np.max(probability_map))
    prob_mean = float(np.mean(probability_map))
    prob_std = float(np.std(probability_map))

    prob_p95 = float(np.percentile(probability_map, 95))
    prob_p99 = float(np.percentile(probability_map, 99))
    prob_p999 = float(np.percentile(probability_map, 99.9))

    pixels_ge_01 = int(np.sum(probability_map >= 0.10))
    pixels_ge_025 = int(np.sum(probability_map >= 0.25))
    predicted_pixels = int(np.sum(predicted_mask))

    total_pixels = TARGET_SIZE[0] * TARGET_SIZE[1]

    predicted_percentage = (
        predicted_pixels / total_pixels
    ) * 100.0


# ============================================================
# FOUR REQUIRED OUTPUTS — SPLIT DASHBOARD
# Left: input document + prediction summary
# Right: all three required visualizations
# ============================================================
st.markdown('<div class="section-title">Localization Results</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Review the prediction summary and the four required localization outputs.</div>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.88, 1.55], gap="large")

# ============================================================
# LEFT SIDE — INPUT + PREDICTION SUMMARY
# ============================================================
with left_col:
    st.markdown(
        '<div class="output-card">'
        '<div class="output-title">1. Input Document</div>'
        '<div class="output-description">Original document provided to the model.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.image(pil_img, use_container_width=True)

    if prob_max >= THRESHOLD:
        conclusion_title = "TAMPERING DETECTED"
        conclusion_text = (
            f"The probability map contains pixels at or above the "
            f"{THRESHOLD:.2f} decision threshold, producing positive "
            "pixels in the binary tamper mask."
        )
        conclusion_class = "detected"
    else:
        conclusion_title = "NO TAMPERING DETECTED"
        conclusion_text = (
            f"No pixel reached the {THRESHOLD:.2f} decision threshold. "
            "The continuous probability map may still show a sub-threshold response."
        )
        conclusion_class = "clear"

    st.markdown(
        f"""<div class="result-card" style="margin-top:18px;">
            <div class="result-label">Prediction Summary</div>
            <div class="result-status {conclusion_class}">{conclusion_title}</div>
            <div class="result-description">{conclusion_text}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    summary1, summary2 = st.columns(2)
    with summary1:
        st.metric("Maximum Probability", f"{prob_max:.5f}")
        st.metric("Tampered Pixels", f"{predicted_pixels:,}")
    with summary2:
        st.metric("Mean Probability", f"{prob_mean:.5f}")
        st.metric("Predicted Area", f"{predicted_percentage:.2f}%")

    st.caption(f"Decision threshold: probability ≥ {THRESHOLD:.2f}")

    st.markdown(
        f"""<div class="info-card" style="margin-top:14px;">
            <div class="info-title">Quick Interpretation</div>
            <div class="info-row"><span class="info-key">Minimum</span><span class="info-value">{prob_min:.5f}</span></div>
            <div class="info-row"><span class="info-key">Maximum</span><span class="info-value">{prob_max:.5f}</span></div>
            <div class="info-row"><span class="info-key">Mean</span><span class="info-value">{prob_mean:.5f}</span></div>
            <div class="info-row"><span class="info-key">Threshold</span><span class="info-value">{THRESHOLD:.2f}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

# ============================================================
# RIGHT SIDE — ALL THREE REQUIRED MAPS
# ============================================================
with right_col:
    # 2. STANDARD PROBABILITY MAP — SMALLER
    st.markdown(
        '<div class="output-card">'
        '<div class="output-title">2. Tamper Probability Map</div>'
        '<div class="output-description">Probability response on the fixed 0–1 scale.</div>'
        '<span class="map-note">Fixed probability scale: 0 to 1</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # EXISTING PROBABILITY MAP GENERATION — UNCHANGED
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    im = ax.imshow(probability_map, cmap="hot", vmin=0.0, vmax=1.0)
    ax.set_title("Predicted Tamper Probability Map (Scale: 0–1)", fontsize=10, pad=8)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # 3. CONTINUOUS PROBABILITY MAP — LARGE
    st.markdown(
        '<div class="output-card" style="margin-top:18px;">'
        '<div class="output-title">3. Continuous Probability Map</div>'
        '<div class="output-description">Detailed spatial response using the existing full dynamic display range.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # EXISTING CONTINUOUS MAP GENERATION — UNCHANGED
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    p_min = float(np.min(probability_map))
    p_max = float(np.max(probability_map))
    vmin = p_min
    if (p_max - p_min) > 1e-6:
        vmax = p_max
    else:
        vmax = p_min + 1e-3
    im_full = ax.imshow(probability_map, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(
        f"Continuous Probability Map (Full Dynamic Range: [{p_min:.5f}, {p_max:.5f}])",
        fontsize=11,
        pad=9,
    )
    ax.axis("off")
    cbar_full = plt.colorbar(im_full, ax=ax, fraction=0.046, pad=0.04)
    cbar_full.ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # 4. PREDICTED BINARY TAMPER MASK — LARGE
    st.markdown(
        '<div class="output-card" style="margin-top:18px;">'
        '<div class="output-title">4. Predicted Binary Tamper Mask</div>'
        '<div class="output-description">Final pixel-level localization from the existing 0.5 threshold.</div>'
        '<span class="map-note">Decision threshold: probability ≥ 0.5</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # EXISTING BINARY MASK GENERATION — UNCHANGED
    mask_display = (predicted_mask * 255).astype(np.uint8)
    st.image(
        mask_display,
        caption="Predicted Binary Tamper Mask (Probability ≥ 0.5)",
        clamp=True,
        use_container_width=True,
    )

# ============================================================
# PROBABILITY DETAILS — SECONDARY INFORMATION
# ============================================================
st.markdown('<div class="section-title" style="margin-top:28px;">Probability Details</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Additional statistics from the same probability map used above.</div>', unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Minimum Probability", f"{prob_min:.5f}")
with s2:
    st.metric("Standard Deviation", f"{prob_std:.5f}")
with s3:
    st.metric("95th Percentile", f"{prob_p95:.5f}")
with s4:
    st.metric("99th Percentile", f"{prob_p99:.5f}")

p1, p2, p3 = st.columns(3)
with p1:
    st.metric("99.9th Percentile", f"{prob_p999:.5f}")
with p2:
    st.metric("Pixels ≥ 0.10", f"{pixels_ge_01:,}")
with p3:
    st.metric("Pixels ≥ 0.25", f"{pixels_ge_025:,}")


# ============================================================
# TECHNICAL INFORMATION — SECONDARY / BOTTOM
# ============================================================

with st.expander("Model Information"):
    checkpoint_epoch = checkpoint.get("epoch", "Unknown")
    best_val_f1 = checkpoint.get(
        "best_val_f1",
        checkpoint.get("val_stats", {}).get("val_f1", "Unknown")
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-row"><span class="info-key">Architecture</span><span class="info-value">RGB + DCT + SRM TamperDoc</span></div>
            <div class="info-row"><span class="info-key">Checkpoint</span><span class="info-value">{CHECKPOINT}</span></div>
            <div class="info-row"><span class="info-key">Checkpoint Epoch</span><span class="info-value">{checkpoint_epoch}</span></div>
            <div class="info-row"><span class="info-key">Best Validation F1</span><span class="info-value">{best_val_f1}</span></div>
            <div class="info-row"><span class="info-key">Input</span><span class="info-value">512 × 512 × 3</span></div>
            <div class="info-row"><span class="info-key">Output</span><span class="info-value">512 × 512</span></div>
            <div class="info-row"><span class="info-key">Decision Threshold</span><span class="info-value">0.50</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="footer">TamperDoc · Document Tamper Localization · RGB + DCT + SRM</div>',
    unsafe_allow_html=True,
)
