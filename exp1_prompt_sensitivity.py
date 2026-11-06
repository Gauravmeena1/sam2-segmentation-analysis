"""
exp1_prompt_sensitivity.py
Experiment 1: How sensitive is SAM 2 to the *type* of prompt?

For every image we run four prompt variants through the SAM 2 image predictor:
    1point      - a single foreground click
    3points     - the click plus two nearby foreground clicks
    box         - a bounding box
    box+point   - the box plus the foreground click

For each variant we record:
    - the model's predicted IoU ("score" = SAM 2's own confidence)
    - the predicted mask area (fraction of the image)
    - an overlay image saved to disk
We then measure how much the chosen mask CHANGES between prompt types
(pairwise IoU vs the box mask = "stability"), and, if ground-truth masks are
supplied, the TRUE IoU of each variant.

Works without ground truth (confidence + stability).
Pass --gt_dir for a stronger quantitative IoU comparison.

Prompts default to the image centre. For best results give a per-image prompt
via --prompts_json:
    { "cat.jpg": {"point": [x, y], "box": [x0, y0, x1, y1]} }
(any field may be omitted; pixel coords, origin top-left)

Run from inside the cloned `sam2/` repo, e.g.:
    python exp1_prompt_sensitivity.py --images_dir data/prompt_images \
        --out_dir results/exp1 --model_size large
"""

import os
import csv
import json
import argparse
import numpy as np
import torch

from utils import (build_image_predictor, inference_context, load_image,
                   save_overlay, mask_iou)


def default_prompts(h, w, override=None):
    override = override or {}
    point = np.array(override.get("point", [w / 2, h / 2]), dtype=float)
    box = np.array(override.get("box", [w * 0.25, h * 0.25, w * 0.75, h * 0.75]), dtype=float)
    return point, box


def jitter_points(point, h, w, n=2, frac=0.08):
    """Extra foreground points near `point` for the multi-point variant."""
    pts = [point.tolist()]
    for _ in range(n):
        dx = (np.random.rand() - 0.5) * 2 * frac * w
        dy = (np.random.rand() - 0.5) * 2 * frac * h
        pts.append([float(np.clip(point[0] + dx, 0, w - 1)),
                    float(np.clip(point[1] + dy, 0, h - 1))])
    return np.array(pts, dtype=float)


def predict_best(predictor, point=None, labels=None, box=None, multimask=True):
    masks, scores, _ = predictor.predict(
        point_coords=point, point_labels=labels, box=box, multimask_output=multimask)
    i = int(np.argmax(scores))
    return masks[i].astype(bool), float(scores[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", default="results/exp1")
    ap.add_argument("--gt_dir", default=None, help="optional binary GT masks (same name, .png)")
    ap.add_argument("--prompts_json", default=None)
    ap.add_argument("--model_size", default="large",
                    choices=["tiny", "small", "base_plus", "large"])
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    sidecar = json.load(open(args.prompts_json)) if args.prompts_json else {}

    predictor, device = build_image_predictor(args.model_size, args.checkpoint_dir)
    print(f"[exp1] SAM 2 ({args.model_size}) on {device}")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted(f for f in os.listdir(args.images_dir) if f.lower().endswith(exts))
    if not files:
        raise SystemExit(f"No images found in {args.images_dir}")

    rows = []
    for fname in files:
        img = load_image(os.path.join(args.images_dir, fname))
        h, w = img.shape[:2]
        point, box = default_prompts(h, w, sidecar.get(fname))
        pts3 = jitter_points(point, h, w, n=2)
        p1 = point[None, :]

        gt = None
        if args.gt_dir:
            gp = os.path.join(args.gt_dir, os.path.splitext(fname)[0] + ".png")
            if os.path.isfile(gp):
                gt = np.array(load_image(gp)[..., 0] > 127, dtype=bool)

        with torch.inference_mode(), inference_context(device):
            predictor.set_image(img)
            variants = {
                "1point":    predict_best(predictor, p1, np.array([1]), multimask=True),
                "3points":   predict_best(predictor, pts3, np.ones(len(pts3)), multimask=True),
                "box":       predict_best(predictor, box=box, multimask=False),
                "box+point": predict_best(predictor, p1, np.array([1]), box=box, multimask=False),
            }

        ref = variants["box"][0]  # stability reference
        stem = os.path.splitext(fname)[0]
        for name, (mask, score) in variants.items():
            out_png = os.path.join(args.out_dir, "overlays", f"{stem}__{name}.png")
            save_overlay(
                img, mask, out_png,
                points=(pts3 if name == "3points" else (p1 if "point" in name else None)),
                box=(box if "box" in name else None),
                title=f"{name}  (pred IoU {score:.2f})")
            row = {"image": fname, "variant": name,
                   "pred_score": round(score, 4),
                   "area_frac": round(float(mask.sum()) / (h * w), 4),
                   "iou_vs_box": round(mask_iou(mask, ref), 4)}
            if gt is not None:
                row["iou_gt"] = round(mask_iou(mask, gt), 4)
            rows.append(row)
        print(f"  {fname}: " + ", ".join(f"{n}={v[1]:.2f}" for n, v in variants.items()))

    # CSV
    csv_path = os.path.join(args.out_dir, "exp1_results.csv")
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=keys)
        w_.writeheader()
        w_.writerows(rows)
    print(f"[exp1] wrote {csv_path}")

    _summary_plots(rows, args.out_dir)
    print(f"[exp1] done -> {args.out_dir}")


def _summary_plots(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["1point", "3points", "box", "box+point"]

    def mean_by(key):
        return {v: (np.mean([r[key] for r in rows if r["variant"] == v and key in r])
                    if any(r["variant"] == v and key in r for r in rows) else 0.0)
                for v in order}

    metrics = [("pred_score", "Mean predicted IoU (confidence)")]
    if any("iou_gt" in r for r in rows):
        metrics.append(("iou_gt", "Mean IoU vs ground truth"))
    metrics.append(("iou_vs_box", "Agreement with box prompt (stability)"))

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    for ax, (key, label) in zip(axes, metrics):
        d = mean_by(key)
        ax.bar(list(d.keys()), list(d.values()), color="#3b82f6")
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_summary.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
