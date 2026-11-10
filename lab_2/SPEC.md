# SPEC: Backdoor target-class detector (P2)

## 1. Goal
For each of the 200 test models and each of the 20 classes, output a suspicion score that the class is the target of a backdoor. The metric is ROC-AUC pooled over all 200×20 pairs (see `task.md`), so scores must be comparable across models. Beat the Neural Cleanse baseline (public AUC 0.901) with margin, because the private half contains more hard attacks.

Deadline: 2026-11-23 16:00. Limit: 10 submissions/day.

## 2. Decisions
| Topic | Decision |
|---|---|
| Core approach | Trigger inversion, Neural Cleanse + K-Arm style. No meta-classifier (MNTD is not viable with 400 images and no base model/training data). |
| Compute budget | About 2 h per full run over 200 models on an RTX 3080 (10 GB), roughly 35 s/model. |
| Trigger forms | Patch (mask+pattern), blended/additive perturbation, WaNet-like warp. |
| Targeted (source-specific) backdoors | Explicit per-source scans. |
| Cross-model comparability | Absolute score + population calibration across the 200 test models. |
| Train models (6) | Smoke test and rough calibration only (3 positive pairs: far too few to fit anything). |
| Self-made poisoned models | Not required. Optional stretch only. |
| Real validation | Public leaderboard (submit the full pipeline; no ablation studies). |

## 3. Inputs
- `example/model.py`: `load(path, device)`; the net is ResNet-18 (no maxpool) with a cosine classifier head, input (N,3,64,64) in [0,1], normalisation inside.
- `example/clean/` (400 images, 20 per class, `labels.csv`), `example/classes.txt`.
- `example/train/` (6 labelled models), `test/models/*.pt` (200 models).

## 4. Method

### 4.1 Data and model handling
- Load the 400 clean images into one GPU tensor in [0,1]. Load models with `load(path, device="cuda")`, freeze all parameters, use autocast and `channels_last`.
- Split the clean images with a fixed seed, stratified by class, into an optimisation split and a held-out split. Default 75/25 (15 and 5 images per class; 100 held-out images in total). The split fraction is a config value. The same split is used for every model, so scores stay comparable across models.
- In the all-class scan (4.2), ASR is measured on held-out images.

### 4.2 Batched all-class scan (K-Arm style)
Optimise all 20 target classes in parallel. Trigger parameters have shape (20, …) and each step forwards 20×B non-target images (B ≈ 32–64) in one batch. Per form, minimise CE toward the target plus a size/smoothness penalty, with an adaptive λ (NC-style) so the trigger just reaches ASR ≈ 0.9 on the optimisation split.

- **A. Patch:** x' = (1−m)·x + m·p, m = sigmoid(a), penalty L1(m).
- **B. Blended/additive:** x' = clip(x + δ), smooth/full-image δ, penalty L2/L∞ of δ (also covers SIG-like triggers).
- **C. Warp:** x' = grid_sample(x, id + flow), low-resolution flow upsampled, penalty on flow magnitude + TV.

Budget: about 100–150 Adam steps per form. K-Arm-style: a short first pass over all classes, then the remaining steps only on the top-K (≈ 5) classes per form.

Recorded per (model, class, form): trigger size at the ASR target (L1 of mask / L2 of δ / flow magnitude), ASR at a fixed small budget, final loss.

### 4.3 Source-specific (targeted) scan
- For each (source s, target t), optimise a trigger on the source-s images only (batched over the 380 pairs). A source class has only 20 images in total, so use all 20 for optimisation and no held-out ASR. The signal is the trigger size needed to flip the source class.
- Check or penalise that the trigger does not move other-class images, since targeted backdoors leave other classes alone by design.
- If this scan overfits in practice, fall back to 2-fold cross-fitting (optimise on 10, evaluate on the other 10, and swap).
- Per-target score = max over sources of the calibrated pair score. Run for all targets if the time budget allows; otherwise for the top-K from 4.2 plus a random sample, decided from timing measurements.

### 4.4 Scores and comparability
- Take the log of each raw statistic, then robust z-score (median/MAD) per (class, form) across the 200 test models. The reference is the population, not the within-model spread, so clean models do not produce an artificial "outlier" class.
- Also compute the classic NC within-model MAD anomaly index as an extra feature.
- Final score per (model, class) = max over forms (and the targeted scan) of the calibrated z, or a small fixed-weight combination. Fusion stays low-capacity (hand-set or tiny logistic regression).
- All values finite, and no artificial ties from clipping.

## 5. Validation
1. Smoke test on the 6 train models: backdoored models rank the target class high, clean models score lower, scale is plausible, runtime per model is measured.
2. Public leaderboard: submit the full pipeline and check the AUC against 0.901. No ablation studies. Prefer principled changes over public-half tuning, because private is harder.
3. Optional stretch: plant known triggers into clean models by short fine-tuning on the 400 images to measure recall per trigger type.

## 6. Timeline (from 2026-11-10)
1. Days 1–2: loader, patch-form batched scan, smoke test, first submission (target: beat 0.901).
2. Days 3–5: blend and warp forms, population calibration, timing profile (≤ 2 h per run).
3. Days 6–8: targeted scan, integrate into the full pipeline.
4. Days 9–12: freeze the pipeline, final full run, submit; keep a buffer before the deadline.

## 7. Deliverables
- Code under `src/lab_2/` (the previous implementation is ignored; check the layout before adding): data/model loading, scan forms, targeted scan, scoring/calibration, CSV writer.
- CLI entrypoint (`lab_2 = lab_2.main:main` in `pyproject.toml`) that scans a model directory and caches per-model raw statistics, so scoring can be re-tuned without rescanning.
- `submission.csv` with columns `id,c00..c19`.

## 8. Acceptance checks
- 6 train models: backdoored models rank their target class highest or near-highest, clean models score lower.
- One-model timing extrapolates to ≤ ~2 h for 200 models.
- CSV validity: 200 unique ids matching `test/models/*.pt`, 20 finite columns, no nan/inf.
- Public AUC above 0.901.
