import os
import io
import lmdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(r"C:\projects\TamperDoc\dataset\DocTamper")

TRAIN_DB = DATASET_DIR / "DocTamperV1-TrainingSet"

OUTPUT_DIR = Path(r"C:\projects\TamperDoc\results\eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of samples used for numerical EDA
# Set to None to process the complete training set.
MAX_SAMPLES = None


# ============================================================
# OPEN LMDB
# ============================================================

print("=" * 70)
print("DocTamper EDA")
print("=" * 70)

print(f"Dataset: {TRAIN_DB}")
print(f"Output : {OUTPUT_DIR}")

env = lmdb.open(
    str(TRAIN_DB),
    readonly=True,
    lock=False,
    readahead=False
)


# ============================================================
# GET ALL IMAGE KEYS
# ============================================================

with env.begin() as txn:

    image_keys = []

    cursor = txn.cursor()

    for key, value in cursor:

        if key.startswith(b"image-"):
            image_keys.append(key)

image_keys = sorted(image_keys)

if MAX_SAMPLES is not None:
    image_keys = image_keys[:MAX_SAMPLES]

print(f"\nNumber of image samples analysed: {len(image_keys)}")


# ============================================================
# STORAGE FOR EDA STATISTICS
# ============================================================

image_widths = []
image_heights = []

jpeg_sizes = []

tampered_pixels = []
tampered_percentages = []

total_tampered_pixels = 0
total_background_pixels = 0

sample_data = []


# ============================================================
# READ DATASET
# ============================================================

with env.begin() as txn:

    for i, image_key in enumerate(image_keys):

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image_bytes = txn.get(image_key)

        if image_bytes is None:
            print(f"WARNING: Missing image: {image_key}")
            continue

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_np = np.asarray(image)

        height, width = image_np.shape[:2]

        image_widths.append(width)
        image_heights.append(height)

        jpeg_sizes.append(len(image_bytes))


        # ----------------------------------------------------
        # GROUND-TRUTH MASK
        # ----------------------------------------------------

        index = image_key.decode("utf-8").split("-")[1]

        label_key = f"label-{index}".encode("utf-8")

        mask_bytes = txn.get(label_key)

        if mask_bytes is None:
            print(f"WARNING: Missing mask: {label_key}")
            continue

        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

        mask_np = np.asarray(mask)

        # Binary mask
        mask_binary = mask_np > 0

        tampered_count = int(np.count_nonzero(mask_binary))

        total_pixels = mask_binary.size

        background_count = total_pixels - tampered_count

        tampered_percentage = (
            tampered_count / total_pixels
        ) * 100.0


        # ----------------------------------------------------
        # STORE STATISTICS
        # ----------------------------------------------------

        tampered_pixels.append(tampered_count)

        tampered_percentages.append(tampered_percentage)

        total_tampered_pixels += tampered_count
        total_background_pixels += background_count


        # ----------------------------------------------------
        # STORE FIRST FEW SAMPLES FOR VISUALIZATION
        # ----------------------------------------------------

        if len(sample_data) < 6:

            sample_data.append(
                (
                    image_np,
                    mask_binary,
                    image_key.decode("utf-8")
                )
            )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (i + 1) % 1000 == 0:

            print(
                f"Processed {i + 1}/{len(image_keys)} samples..."
            )


env.close()


# ============================================================
# BASIC DATASET STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

print(f"Images analysed          : {len(image_widths)}")

print(
    f"Image width             : "
    f"{min(image_widths)} - {max(image_widths)} pixels"
)

print(
    f"Image height            : "
    f"{min(image_heights)} - {max(image_heights)} pixels"
)

print(
    f"Unique resolutions      : "
    f"{len(set(zip(image_widths, image_heights)))}"
)

print(
    f"JPEG size (KB)          : "
    f"{min(jpeg_sizes)/1024:.2f} - "
    f"{max(jpeg_sizes)/1024:.2f}"
)

print(
    f"Average JPEG size      : "
    f"{np.mean(jpeg_sizes)/1024:.2f} KB"
)


# ============================================================
# TAMPER STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("GROUND-TRUTH TAMpering STATISTICS")
print("=" * 70)

print(
    f"Total tampered pixels   : "
    f"{total_tampered_pixels:,}"
)

print(
    f"Total background pixels : "
    f"{total_background_pixels:,}"
)

total_pixels = (
    total_tampered_pixels +
    total_background_pixels
)

overall_tamper_percentage = (
    total_tampered_pixels /
    total_pixels
) * 100.0

print(
    f"Overall tampered pixels : "
    f"{overall_tamper_percentage:.4f}%"
)

print(
    f"Minimum tampered area   : "
    f"{min(tampered_percentages):.4f}%"
)

print(
    f"Maximum tampered area   : "
    f"{max(tampered_percentages):.4f}%"
)

print(
    f"Average tampered area   : "
    f"{np.mean(tampered_percentages):.4f}%"
)

print(
    f"Median tampered area    : "
    f"{np.median(tampered_percentages):.4f}%"
)


# ============================================================
# EDA 1
# SAMPLE IMAGE + GROUND-TRUTH MASK
# ============================================================

if len(sample_data) > 0:

    fig, axes = plt.subplots(
        len(sample_data),
        2,
        figsize=(10, 4 * len(sample_data))
    )

    if len(sample_data) == 1:
        axes = np.array([axes])

    for i, (image, mask, sample_id) in enumerate(sample_data):

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(
            f"{sample_id} - Original Document"
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title(
            f"{sample_id} - Ground-Truth Tamper Mask"
        )
        axes[i, 1].axis("off")

    plt.tight_layout()

    path = OUTPUT_DIR / "eda_1_sample_images_masks.png"

    plt.savefig(path, dpi=200, bbox_inches="tight")

    plt.close()

    print(f"\nSaved: {path}")


# ============================================================
# EDA 2
# TAMPERRED PIXEL AREA DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    tampered_pixels,
    bins=40
)

plt.xlabel("Number of Tampered Pixels")
plt.ylabel("Number of Images")
plt.title("Distribution of Tampered Pixel Area")

plt.grid(alpha=0.25)

plt.tight_layout()

path = OUTPUT_DIR / "eda_2_tampered_pixel_distribution.png"

plt.savefig(path, dpi=200, bbox_inches="tight")

plt.close()

print(f"Saved: {path}")


# ============================================================
# EDA 3
# TAMPERED AREA PERCENTAGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    tampered_percentages,
    bins=40
)

plt.xlabel("Tampered Area (%)")
plt.ylabel("Number of Images")
plt.title("Distribution of Tampered Area Percentage")

plt.grid(alpha=0.25)

plt.tight_layout()

path = OUTPUT_DIR / "eda_3_tampered_area_percentage.png"

plt.savefig(path, dpi=200, bbox_inches="tight")

plt.close()

print(f"Saved: {path}")


# ============================================================
# EDA 4
# IMAGE RESOLUTION DISTRIBUTION
# ============================================================

resolutions = [
    f"{w}×{h}"
    for w, h in zip(image_widths, image_heights)
]

unique_resolutions, resolution_counts = np.unique(
    resolutions,
    return_counts=True
)

order = np.argsort(resolution_counts)[::-1]

unique_resolutions = unique_resolutions[order]
resolution_counts = resolution_counts[order]

plt.figure(figsize=(10, 6))

plt.bar(
    unique_resolutions,
    resolution_counts
)

plt.xlabel("Image Resolution")
plt.ylabel("Number of Images")
plt.title("Distribution of Image Resolutions")

plt.xticks(rotation=45)

plt.tight_layout()

path = OUTPUT_DIR / "eda_4_resolution_distribution.png"

plt.savefig(path, dpi=200, bbox_inches="tight")

plt.close()

print(f"Saved: {path}")


# ============================================================
# SAVE NUMERICAL SUMMARY
# ============================================================

summary_path = OUTPUT_DIR / "eda_summary.txt"

with open(summary_path, "w") as f:

    f.write("DocTamper EDA SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Images analysed: {len(image_widths)}\n"
    )

    f.write(
        f"Minimum image width: {min(image_widths)}\n"
    )

    f.write(
        f"Maximum image width: {max(image_widths)}\n"
    )

    f.write(
        f"Minimum image height: {min(image_heights)}\n"
    )

    f.write(
        f"Maximum image height: {max(image_heights)}\n"
    )

    f.write(
        f"Unique resolutions: "
        f"{len(set(zip(image_widths, image_heights)))}\n"
    )

    f.write(
        f"Average JPEG size: "
        f"{np.mean(jpeg_sizes)/1024:.2f} KB\n"
    )

    f.write(
        f"Total tampered pixels: "
        f"{total_tampered_pixels:,}\n"
    )

    f.write(
        f"Total background pixels: "
        f"{total_background_pixels:,}\n"
    )

    f.write(
        f"Overall tampered percentage: "
        f"{overall_tamper_percentage:.4f}%\n"
    )

    f.write(
        f"Minimum tampered area: "
        f"{min(tampered_percentages):.4f}%\n"
    )

    f.write(
        f"Maximum tampered area: "
        f"{max(tampered_percentages):.4f}%\n"
    )

    f.write(
        f"Average tampered area: "
        f"{np.mean(tampered_percentages):.4f}%\n"
    )

    f.write(
        f"Median tampered area: "
        f"{np.median(tampered_percentages):.4f}%\n"
    )

print(f"Saved: {summary_path}")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(f"All outputs are in:")
print(OUTPUT_DIR)