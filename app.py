import os
import sys

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as TF


# ============================================================
# PROJECT CONFIGURATION
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
    layout="wide"
)


st.title("Document Tamper Localization")

st.write(
    "Upload a 512×512 document image and run the trained "
    "TamperDoc model to generate pixel-level tamper "
    "probability maps and a predicted binary tamper mask."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():
    model, checkpoint = load_model(CHECKPOINT)
    model.eval()
    return model, checkpoint


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Document Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    try:
        pil_img = Image.open(uploaded_file).convert("RGB")

    except Exception as e:
        st.error(f"Unable to open image: {e}")
        st.stop()


    # --------------------------------------------------------
    # Display uploaded image
    # --------------------------------------------------------

    st.subheader("Uploaded Document")

    st.image(
        pil_img,
        width=500
    )


    # --------------------------------------------------------
    # Check image dimensions
    # --------------------------------------------------------

    if pil_img.size != TARGET_SIZE:

        st.error(
            f"Invalid image size: {pil_img.size}. "
            f"The model requires exactly "
            f"{TARGET_SIZE[0]}×{TARGET_SIZE[1]} pixels."
        )

        st.stop()


    # ========================================================
    # RUN DETECTION
    # ========================================================

    if st.button("Run Tamper Detection", type="primary"):

        with st.spinner("Running TamperDoc model..."):

            # ------------------------------------------------
            # 1. Convert image to tensor
            # ------------------------------------------------

            raw_x = TF.to_tensor(pil_img).unsqueeze(0)


            # ------------------------------------------------
            # 2. ImageNet normalization for RGB branch
            # ------------------------------------------------

            norm_x = TF.normalize(
                raw_x.clone(),
                mean=NORMALIZE_MEAN,
                std=NORMALIZE_STD
            )


            # ------------------------------------------------
            # 3. Load trained model
            # ------------------------------------------------

            try:

                model, checkpoint = get_model()

            except Exception as e:

                st.error(
                    f"Unable to load model checkpoint: {e}"
                )

                st.stop()


            # ------------------------------------------------
            # 4. Actual model inference
            # ------------------------------------------------

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


            # ------------------------------------------------
            # 5. Convert outputs to NumPy
            # ------------------------------------------------

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


            # ------------------------------------------------
            # 6. Sanity check
            # ------------------------------------------------

            if (
                np.isnan(logits_map).any()
                or np.isinf(logits_map).any()
            ):

                st.error(
                    "Model output contains NaN or Inf values."
                )

                st.stop()


            # =================================================
            # COMPUTE STATISTICS
            # =================================================

            prob_min = float(
                np.min(probability_map)
            )

            prob_max = float(
                np.max(probability_map)
            )

            prob_mean = float(
                np.mean(probability_map)
            )

            prob_std = float(
                np.std(probability_map)
            )


            # Percentiles
            prob_p95 = float(
                np.percentile(
                    probability_map,
                    95
                )
            )

            prob_p99 = float(
                np.percentile(
                    probability_map,
                    99
                )
            )

            prob_p999 = float(
                np.percentile(
                    probability_map,
                    99.9
                )
            )


            # Threshold counts
            pixels_ge_01 = int(
                np.sum(
                    probability_map >= 0.10
                )
            )

            pixels_ge_025 = int(
                np.sum(
                    probability_map >= 0.25
                )
            )

            predicted_pixels = int(
                np.sum(predicted_mask)
            )


            total_pixels = (
                TARGET_SIZE[0]
                * TARGET_SIZE[1]
            )


            predicted_percentage = (
                predicted_pixels
                / total_pixels
            ) * 100.0


        st.success(
            "Tamper detection completed."
        )


        # ====================================================
        # MODEL OUTPUTS
        # ====================================================

        st.subheader("Model Output")


        # ====================================================
        # ROW 1 — INPUT + STANDARD PROBABILITY MAP
        # ====================================================

        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # 1. Input Document
        # ----------------------------------------------------

        with col1:

            st.write("### 1. Input Document")

            st.image(
                pil_img,
                use_container_width=True
            )


        # ----------------------------------------------------
        # 2. Standard Probability Map
        # ----------------------------------------------------

        with col2:

            st.write(
                "### 2. Tamper Probability Map"
            )

            st.caption(
                "Fixed probability scale: 0 to 1"
            )


            fig, ax = plt.subplots(
                figsize=(6, 6)
            )


            im = ax.imshow(
                probability_map,
                cmap="hot",
                vmin=0.0,
                vmax=1.0
            )


            ax.set_title(
                "Predicted Tamper Probability Map "
                "(Scale: 0–1)",
                fontsize=10,
                pad=10
            )


            ax.axis("off")


            cbar = plt.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )


            cbar.ax.tick_params(
                labelsize=9
            )


            plt.tight_layout()


            st.pyplot(
                fig,
                use_container_width=True
            )


            plt.close(fig)


        # ====================================================
        # ROW 2 — CONTINUOUS FULL-RANGE MAP
        # ====================================================

        st.subheader(
            "3. Continuous Probability Map"
        )


        st.caption(
            "The color scale is adjusted to the actual "
            "minimum and maximum probability produced "
            "by the model for this image."
        )


        fig, ax = plt.subplots(
            figsize=(8, 8)
        )


        # ----------------------------------------------------
        # Same dynamic-range logic as demo_inference.py
        # ----------------------------------------------------

        p_min = float(
            np.min(probability_map)
        )

        p_max = float(
            np.max(probability_map)
        )


        vmin = p_min

        if (p_max - p_min) > 1e-6:
            vmax = p_max
        else:
            vmax = p_min + 1e-3


        im_full = ax.imshow(
            probability_map,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax
        )


        ax.set_title(
            f"Continuous Probability Map "
            f"(Full Dynamic Range: "
            f"[{p_min:.5f}, {p_max:.5f}])",
            fontsize=12,
            pad=10
        )


        ax.axis("off")


        cbar_full = plt.colorbar(
            im_full,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )


        cbar_full.ax.tick_params(
            labelsize=9
        )


        plt.tight_layout()


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


        # ====================================================
        # ROW 3 — BINARY MASK
        # ====================================================

        st.subheader(
            "4. Predicted Binary Tamper Mask"
        )


        st.write(
            f"Decision threshold: "
            f"probability ≥ {THRESHOLD}"
        )


        mask_display = (
            predicted_mask * 255
        ).astype(np.uint8)


        st.image(
            mask_display,
            caption=(
                "Predicted Binary Tamper Mask "
                "(Probability ≥ 0.5)"
            ),
            clamp=True,
            width=600
        )


        # ====================================================
        # DETECTION INFORMATION
        # ====================================================

        st.subheader(
            "5. Detection Information"
        )


        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Minimum Probability",
            f"{prob_min:.5f}"
        )


        c2.metric(
            "Maximum Probability",
            f"{prob_max:.5f}"
        )


        c3.metric(
            "Mean Probability",
            f"{prob_mean:.5f}"
        )


        c4.metric(
            "Predicted Tamper Area",
            f"{predicted_percentage:.2f}%"
        )


        # ====================================================
        # DETAILED STATISTICS
        # ====================================================

        st.subheader(
            "Probability Statistics"
        )


        s1, s2, s3, s4 = st.columns(4)


        s1.metric(
            "Standard Deviation",
            f"{prob_std:.5f}"
        )


        s2.metric(
            "95th Percentile",
            f"{prob_p95:.5f}"
        )


        s3.metric(
            "99th Percentile",
            f"{prob_p99:.5f}"
        )


        s4.metric(
            "99.9th Percentile",
            f"{prob_p999:.5f}"
        )


        st.write(
            f"Pixels with probability ≥ 0.10: "
            f"**{pixels_ge_01:,}**"
        )


        st.write(
            f"Pixels with probability ≥ 0.25: "
            f"**{pixels_ge_025:,}**"
        )


        st.write(
            f"Pixels with probability ≥ 0.50: "
            f"**{predicted_pixels:,}**"
        )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.subheader(
            "Model Interpretation"
        )


        if prob_max >= THRESHOLD:

            st.success(
                f"The model produced probability values "
                f"above the {THRESHOLD:.2f} decision threshold. "
                f"The binary mask therefore contains pixels "
                f"classified as tampered."
            )

        else:

            st.info(
                f"The maximum predicted probability is "
                f"{prob_max:.5f}, which is below the "
                f"{THRESHOLD:.2f} decision threshold. "
                f"Therefore, no pixels are classified as "
                f"tampered in the binary prediction. "
                f"The continuous probability map still "
                f"shows the model's spatial response."
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.subheader(
            "Model Information"
        )


        checkpoint_epoch = checkpoint.get(
            "epoch",
            "Unknown"
        )


        best_val_f1 = checkpoint.get(
            "best_val_f1",
            checkpoint.get(
                "val_stats",
                {}
            ).get(
                "val_f1",
                "Unknown"
            )
        )


        st.write(
            f"Checkpoint: **{CHECKPOINT}**"
        )


        st.write(
            f"Checkpoint Epoch: **{checkpoint_epoch}**"
        )


        st.write(
            f"Best Validation F1: **{best_val_f1}**"
        )


        st.write(
            "Architecture: **RGB + DCT + SRM "
            "TamperDoc**"
        )


        st.write(
            "Output resolution: **512 × 512 pixels**"
        )