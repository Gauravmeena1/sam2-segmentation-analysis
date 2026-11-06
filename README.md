# SAM 2 Prompt Sensitivity and Failure Cases

Computer Vision final project. Tests how [SAM 2](https://github.com/facebookresearch/sam2) (Ravi et al., ICLR 2025) reacts to different prompt types and where it fails, using the SAM 2.1 Hiera-Large checkpoint with no fine-tuning.

## Experiments

1. **Prompt sensitivity:** compares 1 point, 3 points, box and box + point on 18 photos.
2. **Failure cases:** runs SAM 2 on transparent objects, thin structures and tiny objects.

## Setup

Requires Python 3.10+. An NVIDIA GPU is recommended.

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
pip install -r <this repo>/requirements.txt
cd checkpoints && ./download_ckpts.sh && cd ..
```

Copy `utils.py`, `exp1_prompt_sensitivity.py`, `exp2_failure_cases.py` and `data/` into the `sam2/` folder.

## Run

Run from inside the `sam2/` folder.

```bash
python exp1_prompt_sensitivity.py --images_dir data/prompt_images --out_dir results/exp1 --model_size large

python exp2_failure_cases.py --images_dir data/failure_cases --out_dir results/exp2 --model_size large --mode prompt --prompts_json data/prompts.json
```

Without a prompt file, the click is the image centre and the box is the middle 50% of the image, not a tight box around the object.

## Results

Prompt sensitivity, averaged over the 18 photos, from `results/exp1/exp1_results.csv`. Agreement is the mean IoU with the box mask.

| Prompt | Agreement with box |
|---|---|
| 1 point | 0.26 |
| 3 points | 0.36 |
| Box + point | 0.76 |

Mean predicted IoU was 0.81 for 1 point and 0.61 for the box.

Failure cases, with one point prompt per image. Predicted IoU is SAM 2's own score, not measured accuracy, so a high score can still be the wrong object. `data/prompts.json` has the deck click, and the bird click was not saved. These scores match the overlays in `results/exp2/`. A later re-run changed three of them by up to 0.03.

| Image | Predicted IoU |
|---|---|
| Clear glass mug | 0.30 |
| Clear glass tumbler | 0.48 |
| Clear plastic container | 0.09 |
| Clear water bottle | 0.04 |
| Frosted plastic container | 0.38 |
| Frosted water bottle | 0.73 |
| Glass window | 0.96 |
| Tinted green mug | 0.20 |
| Hose sculpture, `.webp` photo | 0.11 |
| Hose sculpture, `.jpg` photo | 0.14 |
| Distant pigeons, click on the bird | 0.93 |
| Distant pigeons, click on the deck | 0.98 |

## Files

```
utils.py                     model loading, overlays, IoU, grids
exp1_prompt_sensitivity.py   experiment 1
exp2_failure_cases.py        experiment 2
data/prompt_images/          18 photos for experiment 1
data/failure_cases/          photos for experiment 2
data/prompts.json            click point for the pigeon photo
results/exp1/                results table and summary chart for experiment 1
results/exp2/                overlays and grids for experiment 2
```
