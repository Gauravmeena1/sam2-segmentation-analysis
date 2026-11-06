"""
utils.py - shared helpers for the SAM 2 analysis project.

Provides:
  - device detection + an inference context (autocast bf16 on CUDA only)
  - a builder for the SAM 2 image predictor (maps model size -> config + checkpoint)
  - visualization helpers (overlay a mask + prompt markers and save to disk)
  - mask IoU and a simple contact-sheet grid

Thin wrappers around the official SAM 2 API: https://github.com/facebookresearch/sam2
"""

import os
import contextlib
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless backend: works over SSH / on a cluster
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


# model size -> (hydra config path, checkpoint filename)
# checkpoints come from:  cd checkpoints && ./download_ckpts.sh
SAM2_VARIANTS = {
    "tiny":      ("configs/sam2.1/sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt"),
    "small":     ("configs/sam2.1/sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large":     ("configs/sam2.1/sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt"),
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def inference_context(device):
    """autocast(bf16) on CUDA; a no-op context elsewhere. Wrap with inference_mode()."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_image(path):
    """Read an image as an HxWx3 uint8 RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def build_image_predictor(model_size="large", checkpoint_dir="checkpoints", device=None):
    """Construct a SAM2ImagePredictor for the requested model size."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if model_size not in SAM2_VARIANTS:
        raise ValueError(f"model_size must be one of {list(SAM2_VARIANTS)}")
    cfg, ckpt_name = SAM2_VARIANTS[model_size]
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run:  cd checkpoints && ./download_ckpts.sh && cd .."
        )
    device = device or get_device()
    sam2_model = build_sam2(cfg, ckpt_path, device=device)
    return SAM2ImagePredictor(sam2_model), device


def mask_iou(a, b):
    """Intersection-over-union of two boolean masks."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def save_overlay(image, mask, out_path, points=None, point_labels=None,
                 box=None, title=None):
    """Save `image` with a translucent `mask` plus any prompt markers."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(image)
    if mask is not None:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.55])  # translucent blue
        ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))
    if box is not None:
        x0, y0, x1, y1 = box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               edgecolor="lime", facecolor="none", lw=2))
    if points is not None and len(points):
        points = np.asarray(points)
        labels = np.asarray(point_labels) if point_labels is not None else np.ones(len(points))
        for (x, y), lab in zip(points, labels):
            ax.scatter(x, y, c="lime" if lab == 1 else "red", marker="*",
                       s=200, edgecolor="white", linewidth=1.2)
    if title:
        ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_color_masks(image, masks, out_path, title=None):
    """Overlay many masks (automatic-mask-generator output) with random colors."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(image)
    rng = np.random.default_rng(0)
    canvas = np.zeros((h, w, 4))
    for m in sorted(masks, key=lambda x: x["area"], reverse=True):  # small masks on top
        canvas[m["segmentation"]] = np.concatenate([rng.random(3), [0.5]])
    ax.imshow(canvas)
    if title:
        ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_grid(image_paths, out_path, cols=4, title=None):
    """Tile saved overlay PNGs into one contact sheet."""
    if not image_paths:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = len(image_paths)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, image_paths):
        ax.imshow(Image.open(p))
        ax.set_title(os.path.basename(p), fontsize=7)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
