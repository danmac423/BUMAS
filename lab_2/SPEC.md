# SPEC: Backdoor target-class detector (P2), simple version

## 1. Goal
For each of the 200 test models and each of the 20 classes, output a suspicion score that the class is the target of a backdoor. The metric is ROC-AUC pooled over all 200×20 pairs (see `task.md`). Beat the Neural Cleanse baseline (public AUC 0.901).

Deadline: 2026-11-23 16:00. Limit: 10 submissions/day.

## 2. Decisions
| Topic | Decision |
|---|---|
| Core approach | Plain Neural Cleanse trigger inversion (mask + pattern) with a K-Arm schedule to save compute. |
| Trigger form | One: `x' = (1-m)·x + m·p`. A full-size, faint mask also covers blended triggers. |
| Score | Neural Cleanse anomaly index of the mask L1 norms, computed within each model. |
| Not included | Warp/blend forms, source-specific scans, population calibration, meta-classifier, self-made poisoned models, held-out splits. |
| Validation | 6 train models as a smoke test, then the public leaderboard. |

Known limitation (accepted for simplicity): source-specific (targeted) backdoors and WaNet-like warps are not modelled explicitly, so they are likely to be partially missed.

## 3. Inputs
- `example/model.py`: `load(path, device)`. ResNet-18, cosine head, input (N,3,64,64) in [0,1].
- `example/clean/` (400 images, `labels.csv`), `example/train/` (6 labelled models), `test/models/*.pt` (200 models).

## 4. Method

### 4.1 Per-class trigger inversion (Neural Cleanse)
For target class `t` learn a mask `m = sigmoid(a)` of shape (1,64,64) and a pattern `p = sigmoid(q)` of shape (3,64,64):

    x' = (1 - m) * x + m * p
    loss_t = CE(net(x'), t) + λ_t * sum(m)

- Images are the clean images whose label is not `t`.
- Frozen model, Adam, bf16 autocast.
- **Adaptive λ** (the NC trick): every few steps, if the attack success rate (ASR) in the last window is ≥ 0.9, raise λ (shrink the mask harder); otherwise lower it. The mask is therefore pushed to the smallest size that still reaches ASR ≈ 0.9.
- Recorded norm for class `t`: the smallest `sum(m)` seen in a window with ASR ≥ 0.9. If the class never reached 0.9, use the final `sum(m)` (large, as it should be).

### 4.2 Batched over classes
All 20 classes are optimised at once. Parameters have shape (20, ...). Each step draws `B` images per class and runs one forward pass of `20·B` images. Per-class losses are summed, so classes do not interact.

### 4.3 K-Arm schedule
1. **Warm-up:** all 20 classes for `S1` steps (default 60).
2. **Focus:** keep the `K` classes (default 5) with the smallest recorded norm and run `S2` more steps (default 100) on them only, with a larger batch per class.
3. The other classes keep their warm-up norm. Because they were ranked lower, they stay higher than the focused ones, which is what the anomaly index needs.

### 4.4 Score
Per model, with `l1[c]` the recorded norm of class `c`:

    med = median(l1);  mad = 1.4826 * median(|l1 - med|)
    score[c] = (med - l1[c]) / mad

A small trigger gives a large positive score (NC anomaly index). Scores are finite; a tiny epsilon guards `mad`.

Raw `(l1, asr)` per model are cached to disk, so scoring can be changed without rescanning.

## 5. Validation
1. Smoke test on the 6 train models: the 3 backdoored models should rank their target class near the top, clean models should score lower, and the time per model should extrapolate to about 2 h or less for 200 models.
2. Submit to the public leaderboard and compare with 0.901.
3. If below 0.901, the first lever is calibration (a population-level z-score of `log l1` across the 200 models). Add it only if needed.

## 6. Deliverables
- `src/lab_2/scan.py` (data loading, inversion, K-Arm), `src/lab_2/scoring.py` (anomaly index, CSV, AUC), `src/lab_2/main.py` (CLI `lab_2`, with subcommands `scan` and `score`).
- `submission.csv` with columns `id,c00..c19`.

## 7. Acceptance checks
- Train smoke test as in 5.1.
- CSV validity: 200 unique ids matching `test/models/*.pt`, 20 finite columns.
- Public AUC above 0.901.
