# Third-party notices

This project builds on the following implementations. Original notices and
licenses remain applicable to the corresponding third-party code; this file
does not relicense that code.

## MACIL

The training, model, and data-management implementation is adapted from
[MACIL](https://github.com/fwu11/MACIL), which in turn acknowledges PILOT.
No license file was found in MACIL's public `main` branch when this release
was prepared. This repository does not assign a new license to MACIL-specific
contributions.

## PILOT

The continual-learning and data-management utilities include code inherited
from [PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT) through MACIL.
PILOT is distributed under the MIT License. Its copyright and complete license
are retained in [licenses/PILOT-MIT.txt](licenses/PILOT-MIT.txt).

Source: <https://raw.githubusercontent.com/LAMDA-CL/LAMDA-PILOT/main/LICENSE>.

## timm / PyTorch Image Models

`models/vit.py` is modified from the Vision Transformer implementation in
[timm](https://github.com/huggingface/pytorch-image-models). The original
Ross Wightman copyright and acknowledgments are retained in the file.
The complete Apache License 2.0 is included in
[licenses/timm-Apache-2.0.txt](licenses/timm-Apache-2.0.txt).

Source: <https://raw.githubusercontent.com/huggingface/pytorch-image-models/v0.6.12/LICENSE>.

## torchvision

The image-loading helpers in `utils/data_manager.py` reference and derive from
[torchvision](https://github.com/pytorch/vision). Its BSD 3-Clause License is
included in
[licenses/torchvision-BSD-3-Clause.txt](licenses/torchvision-BSD-3-Clause.txt).

Source: <https://raw.githubusercontent.com/pytorch/vision/v0.15.2/LICENSE>.

Datasets and pretrained weights are not included in this repository and remain
subject to their respective providers' terms.
