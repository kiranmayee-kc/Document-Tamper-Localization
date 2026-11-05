import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# Import the EXISTING Demo 4 synthesis code.
from demo4_synthesize_2436 import (
    validate_input_image,
    synthesize_tampered_document,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

# Original input
INPUT_IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "demo_inputs",
    "og current bill.jpg"
)

# NEW directory — does not touch demo4_2436
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "demo4v2_2436"
)


# ============================================================
# NEW OUTPUT NAMES
# ============================================================

INTERMEDIATE_PNG = os.path.join(
    OUTPUT_DIR,
    "demo4v2_2436_tampered_intermediate.png"
)

ORIGINAL_PNG = os.path.join(
    OUTPUT_DIR,
    "demo4v2_2436_original.png"
)

GT_MASK_PNG = os.path.join(
    OUTPUT_DIR,
    "demo4v2_2436_gt_mask.png"
)

CHECK_IMAGE_PNG = os.path.join(
    OUTPUT_DIR,
    "demo4v2_2436_comparison.png"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DEMO 4 V2 — SAVE INTERMEDIATE TAMPERED IMAGE")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Load ORIGINAL image
    # --------------------------------------------------------

    print("\n[1/3] Loading original bill...")

    img_rgb = validate_input_image(
        INPUT_IMAGE_PATH
    )

    print(
        f"  [OK] Original image loaded: "
        f"{img_rgb.shape}"
    )


    # --------------------------------------------------------
    # 2. Run EXACT SAME Demo 4 synthesis
    # --------------------------------------------------------

    print(
        "\n[2/3] Running exact Demo 4 synthesis "
        "(1120.00 -> 2436.00)..."
    )

    (
        tampered_rgb,
        gt_mask,
        metadata,
        sauvola_fg
    ) = synthesize_tampered_document(
        img_rgb
    )

    print(
        "  [OK] tampered_rgb generated "
        "in memory."
    )


    # --------------------------------------------------------
    # 3. SAVE NEW FILES ONLY
    # --------------------------------------------------------

    print(
        "\n[3/3] Saving V2 outputs..."
    )

    # Original image
    Image.fromarray(
        img_rgb,
        mode="RGB"
    ).save(
        ORIGINAL_PNG,
        format="PNG"
    )

    # EXACT intermediate tampered_rgb
    # PNG = lossless
    Image.fromarray(
        tampered_rgb,
        mode="RGB"
    ).save(
        INTERMEDIATE_PNG,
        format="PNG"
    )

    # Ground-truth mask
    Image.fromarray(
        gt_mask,
        mode="L"
    ).save(
        GT_MASK_PNG,
        format="PNG"
    )


    # --------------------------------------------------------
    # Create simple side-by-side comparison
    # --------------------------------------------------------

    fig_width = 1024
    fig_height = 512

    comparison = np.concatenate(
        [
            img_rgb,
            tampered_rgb
        ],
        axis=1
    )

    comparison_image = Image.fromarray(
        comparison,
        mode="RGB"
    )

    comparison_image.save(
        CHECK_IMAGE_PNG,
        format="PNG"
    )


    # --------------------------------------------------------
    # Print exact information
    # --------------------------------------------------------

    changed_pixels = np.max(
        np.abs(
            img_rgb.astype(np.int32)
            -
            tampered_rgb.astype(np.int32)
        ),
        axis=-1
    )

    changed_pixel_count = int(
        np.sum(changed_pixels > 0)
    )

    changed_pixel_percentage = (
        changed_pixel_count
        /
        (512 * 512)
    ) * 100.0


    print("\n" + "=" * 70)
    print("V2 EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nChanged pixels: "
        f"{changed_pixel_count:,}"
    )

    print(
        f"Changed pixel percentage: "
        f"{changed_pixel_percentage:.4f}%"
    )

    print("\nNEW FILES:")

    print(
        f"  Original:\n"
        f"    {ORIGINAL_PNG}"
    )

    print(
        f"\n  Intermediate tampered PNG:\n"
        f"    {INTERMEDIATE_PNG}"
    )

    print(
        f"\n  Ground-truth mask:\n"
        f"    {GT_MASK_PNG}"
    )

    print(
        f"\n  Comparison:\n"
        f"    {CHECK_IMAGE_PNG}"
    )

    print("\n" + "=" * 70)

    print(
        "\nIMPORTANT:"
        "\nExisting Demo 4 outputs were NOT modified."
        "\nNo model inference was performed."
        "\nNo existing probability map was overwritten."
    )


if __name__ == "__main__":
    main()