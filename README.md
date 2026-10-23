# M²LE — NeurIPS 2026

Official implementation of **[Modulating Merging Strengths via Joint Loss Estimation for LoRA-based Continual Learning](https://openreview.net/forum?id=zRXaWRbyGH)** (NeurIPS 2026).

Kun Gu, De Cheng, Zhipeng Xu, Lingfeng He, Di Xu, and Nannan Wang.

M²LE combines Hessian Information-guided Adaptive Merging (HIM) with Task Simulation Perturbation (TSP) for replay-free LoRA-based continual learning. This initial release includes the core implementation and an ImageNet-R example with 10 tasks. More configurations and documentation will be added later.

## Environment

The reference environment uses Python 3.9, PyTorch 1.13.1 (CUDA 11.7), torchvision 0.14.1, and timm 0.6.12. Training requires an NVIDIA GPU.

```bash
git clone https://github.com/bbdeeer/M2LE_NeurIPS2026.git
cd M2LE_NeurIPS2026

conda create -n m2le python=3.9 -y
conda activate m2le
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

The ViT-B/16 backbone pretrained on ImageNet-21K is downloaded automatically through timm on the first run. See the [PyTorch installation archive](https://pytorch.org/get-started/previous-versions/#v1131) for other CUDA builds.

## Dataset preparation

Download the preprocessed ImageNet-R dataset following [PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT#datasets), and arrange it as:

```text
datasets/imagenet-r/
├── train/
│   ├── <class_name>/
│   └── ...
└── test/
    ├── <class_name>/
    └── ...
```

Both splits should contain the same 200 class folders. Set `data_path` in `exps/imgr10.json` if your dataset is stored elsewhere. Use the prepared train/test split; the loader does not move or split the original images.

## Train: ImageNet-R (10 tasks)

```bash
python main.py --device 0 --config exps/imgr10.json
```

This setting learns 20 classes per task over 10 tasks. The configuration runs seeds `1993`, `1996`, and `1997` sequentially, with LoRA rank 32, HIM enabled (`use_merge`), and TSP enabled (`use_meta`). Logs for each seed, including final and average accuracy, are saved under `logs/ImageNet_R/10task/`. Change `--device` to select another GPU.

Core code: [`methods/m2le.py`](methods/m2le.py) implements training and HIM merging; [`models/attention.py`](models/attention.py) implements the LoRA attention modules and Fisher collection.

## Citation



## Acknowledgments


