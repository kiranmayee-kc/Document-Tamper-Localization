# TamperDoc: Multi-Branch Document Tamper Localization

TamperDoc is a research codebase for pixel-level document tampering localization on the DocTamper dataset.

## Architecture

The system utilizes a multi-modal multi-branch feature extraction and spatial fusion architecture:
1. **RGB Branch**: Spatial feature extractor using an ImageNet-pretrained ResNet-18 backbone.
2. **DCT Frequency Branch**: Converts the RGB image to luminance, applies a fixed 8×8 DCT-II transform, and processes the resulting frequency representation using a lightweight learnable CNN encoder.
3. **SRM Noise Residual Branch**: Applies three fixed SRM-style high-pass filters independently across the RGB channels to produce nine residual channels, followed by a lightweight learnable CNN encoder.
4. **Spatial Feature Fusion**: Concatenates the 128-channel RGB, DCT, and SRM feature maps to form a 384-channel representation, followed by a learnable projection to 128 channels at 64×64 spatial resolution.
5. **Progressive Localization Decoder**: Progressively upsamples the fused 128-channel 64×64 representation to 512×512 using three stages of bilinear upsampling and convolution.
6. **Pixel-Level Output**: A final 1×1 convolution produces a single-channel 512×512 tamper logit map. Sigmoid converts the logits into a pixel-wise tamper probability map. A threshold of 0.5 is used to generate the final binary tamper localization mask.


The system performs pixel-level tamper localization rather than image-level genuine/forged classification.

## Directory Structure

TamperDoc/
├── checkpoints/
│   └── best_model.pt
├── configs/
│   └── test_subset_10k.json
├── dataset/
│   └── DocTamper/
│       ├── DocTamperV1-TrainingSet/
│       ├── DocTamperV1-Testingset/
│       ├── DocTamperV1-SCD/
│       └── DocTamperV1-FCD/
├── results/
│   ├── evaluation/
│   ├── test_visualizations/
│   └── demo/
├── src/
│   ├── dataset.py
│   ├── rgb_branch.py
│   ├── dct_branch.py
│   ├── srm_branch.py
│   ├── fusion.py
│   ├── localization.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── demo_inference.py
├── requirements.txt
└── README.md

## Task Definition

TamperDoc performs **pixel-level tamper localization**.

The DocTamper dataset used in this project does not provide pristine counterpart images suitable for training a genuine-versus-forged binary classifier. Therefore, the model is not trained as an image-level authenticity classifier.

Instead, the model learns to predict a tamper mask for each document image.

During training, the ground-truth tamper mask provides pixel-level supervision through the combined BCEWithLogits and Dice loss.

During inference, only the document image is provided to the model. 
The ground-truth mask is not part of the forward inference path.