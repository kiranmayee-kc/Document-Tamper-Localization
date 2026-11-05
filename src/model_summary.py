import torch
from torchinfo import summary

from src.model import TamperDocModel


model = TamperDocModel(
    use_rgb=True,
    use_dct=True,
    use_srm=True,
    pretrained_rgb=True
)

summary(
    model,
    input_size=(1, 3, 512, 512),
    depth=4,
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable"
    ]
)