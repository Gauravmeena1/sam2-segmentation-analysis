"""
exp2_failure_cases.py
Experiment 2: Where does SAM 2 break? A failure-case taxonomy.

Organize hard images into category subfolders, e.g.:
    data/failure_cases/
        thin_structures/   (hair, wires, fences, antennae)
        transparent/       (glass, bottles, water)
        camouflage/        (animals/objects blending into the background)
        tiny_objects/      (small/distant objects, dense scenes)

Two modes:
    --mode prompt : prompt the hard object with a point (centre by default, or a
                    per-image point via --prompts_json) and save the mask.
                    Best for "I asked for THIS object and it failed."
    --mode auto   : run the automatic mask generator (segment everything) and
                    colour every mask. Best for over/under-segmentation, missed
                    tiny objects, merged thin structures.

Outputs per category: individual overlays + one contact-sheet grid.

Run from inside the cloned `sam2/` repo, e.g.:
    python exp2_failure_cases.py --images_dir data/failure_cases \
        --out_dir results/exp2 --model_size large --mode prompt
"""

import os
import json
import argparse
import numpy as np
import torch

from utils import (build_image_predictor, inference_context, load_image,
                   save_overlay, save_color_masks, save_grid, get_device, SAM2_VARIANTS)

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _categories(images_dir):
    for category in sorted(os.listdir(images_dir)):
        cdir = os.path.join(images_dir, category)
        if os.path.isdir(cdir):
            files = sorted(f for f in os.listdir(cdir) if f.lower().endswith(EXTS))
            yield category, cdir, files


def run_prompt_mode(args):
    predictor, device = build_image_predictor(args.model_size, args.checkpoint_dir)
    print(f"[exp2/prompt] SAM 2 ({args.model_size}) on {device}")
    sidecar = json.load(open(args.prompts_json)) if args.prompts_json else {}

    for category, cdir, files in _categories(args.images_dir):
        saved = []
        for fname in files:
            img = load_image(os.path.join(cdir, fname))
            h, w = img.shape[:2]
            pt = sidecar.get(f"{category}/{fname}", {}).get("point", [w / 2, h / 2])
            pt = np.array([pt], dtype=float)
            with torch.inference_mode(), inference_context(device):
                predictor.set_image(img)
                masks, scores, _ = predictor.predict(
                    point_coords=pt, point_labels=np.array([1]), multimask_output=True)
            best = masks[int(np.argmax(scores))].astype(bool)
            out_png = os.path.join(args.out_dir, category, fname)
            save_overlay(img, best, out_png, points=pt, point_labels=np.array([1]),
                         title=f"{category}/{fname}  (pred IoU {float(np.max(scores)):.2f})")
            saved.append(out_png)
            print(f"  {category}/{fname}")
        save_grid(saved, os.path.join(args.out_dir, f"{category}__grid.png"),
                  title=f"SAM 2 on: {category}")


def run_auto_mode(args):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    cfg, ckpt_name = SAM2_VARIANTS[args.model_size]
    ckpt = os.path.join(args.checkpoint_dir, ckpt_name)
    device = get_device()
    model = build_sam2(cfg, ckpt, device=device)
    mask_gen = SAM2AutomaticMaskGenerator(model)
    print(f"[exp2/auto] SAM 2 ({args.model_size}) on {device}")

    for category, cdir, files in _categories(args.images_dir):
        saved = []
        for fname in files:
            img = load_image(os.path.join(cdir, fname))
            with torch.inference_mode(), inference_context(device):
                masks = mask_gen.generate(img)
            out_png = os.path.join(args.out_dir, category, fname)
            save_color_masks(img, masks, out_png,
                             title=f"{category}/{fname}  ({len(masks)} masks)")
            saved.append(out_png)
            print(f"  {category}/{fname}: {len(masks)} masks")
        save_grid(saved, os.path.join(args.out_dir, f"{category}__grid.png"),
                  title=f"SAM 2 auto-masks: {category}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", default="results/exp2")
    ap.add_argument("--mode", choices=["prompt", "auto"], default="prompt")
    ap.add_argument("--prompts_json", default=None)
    ap.add_argument("--model_size", default="large",
                    choices=["tiny", "small", "base_plus", "large"])
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    (run_prompt_mode if args.mode == "prompt" else run_auto_mode)(args)
    print(f"[exp2] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
