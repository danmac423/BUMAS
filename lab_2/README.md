# lab_2: backdoor target-class detector (P2)

For every model and each of the 20 classes, output a suspicion score that the class is the target of a
backdoor (ROC-AUC pooled over all model x class pairs). Method: trigger inversion (Neural Cleanse /
K-Arm style) batched over all classes, plus an explicit source-specific scan, calibrated against the
population of test models. See `SPEC.md`.

## Pipeline

1. `lab_2 scan` caches raw per-model statistics (`<cache>/<model id>.npz`, resumable).
   * all-class scan for 3 trigger forms (`patch`, `blend`, `warp`): short pass over all 20 classes, then the
     top-K classes are refined (K-Arm). An adaptive lambda makes every trigger just reach ASR 0.9; the
     recorded statistic is the trigger size at that point (+ held-out ASR).
   * source-specific scan: the all-class patch trigger of each target yields a per-source flip-rate
     matrix; targets with a per-source spike get a pair-level (source, target) optimisation on the source's
     images only.
2. `lab_2 score` calibrates (log size -> robust z-score per class and form against the whole population,
   + within-model NC anomaly index), takes the max over forms / targeted scan and writes `submission.csv`.
   Re-tuning scoring never needs a rescan.

```bash
# data layout: example/ (model.py, clean/, train/) and test/models/*.pt next to it
uv run lab_2 scan  --models test/models --cache cache_scan/test          # ~40 s/model on an RTX 3080
uv run lab_2 score --cache cache_scan/test --models test/models --out submission.csv

# smoke test on the 6 labelled train models, calibrated against the test population
uv run lab_2 scan  --models example/train/models --cache cache_scan/train
uv run lab_2 score --cache cache_scan/train --reference cache_scan/test \
                   --labels example/train/labels.csv --out /tmp/train_sub.csv
```

All knobs live in `src/lab_2/config.py`; override with `--config file.json` (nested, e.g.
`{"p1_steps": 80, "blend": {"enabled": false}}`).

## Optional: plant known triggers to measure recall per trigger type

```bash
uv run lab_2 plant --base example/train/models/<clean>.pt --out cache_synth --spec patch:5 wanet:3 dots:7:2
# kind:target[:source]; a source makes the backdoor source-specific. Writes models/ + labels.csv,
# which `scan` and `score --labels` consume like example/train.
```

Planted models are fitted to the same 400 images the detector scans, so they are easier than the real ones.

## Tests

`uv run pytest` (CPU only).
