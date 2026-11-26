# W11 model training: results, decisions, and resume notes

Updated: 2026-11-23. Repository `main` is at `216acc0` (W10 merged).
All W11 code listed below is currently **uncommitted** on `main`.

## TL;DR (中文)

- 数据缓存、Dataset、训练、四格评估、视频级聚合、subject-level bootstrap 都已跑通。
- 预注册默认配置的 4 格 × 3 seeds 主对比完成：CNN test Macro-F1 0.232–0.268，
  RF-Prior 0.453，差得很明显。
- 5-fold subject-level CV（42 个 train 受试者）显示：更小 head / 更强正则化
  只能把 CV val 提高约 0.03–0.06，幅度在折间噪声内；因此主对比保留默认配置。
- 结论是**严格负结果**：冻结 MobileNetV3-Small 特征在 subject-disjoint 下
  不可作为竞争性模型；瓶颈在特征层，不在数据管道。
- test 集已经在探索阶段被多次评估，当前 test 数字只能标为 exploratory。
- 下次继续：提交代码；等毛冻结 within-subject 划分；跑 4 格 × 3 seeds
  within-subject；整理中期表。微调 backbone（B）留到中期后。

## What was built

| Path | Purpose |
|---|---|
| `src/model/dataset.py` | Real `.npz` reader, BGR→RGB, ImageNet normalization, label mapping, 34 priors, train-only prior normalizer, cached-feature dataset |
| `scripts/extract_features.py` | Frozen MobileNetV3-Small feature cache; supports stratified smoke subsets |
| `src/model/train.py` | 2×2 training, AMP, early stopping, checkpoints, `experiments/runs.csv`, fold-based CV, video-level and bootstrap test metrics |
| `src/model/metrics.py` | Window metrics, video aggregation (mean probability + majority), subject-level bootstrap CI |
| `scripts/smoke_model_data.py` | Dataset contract + small cached-feature overfit smoke test |
| `scripts/tune_frozen_heads.py` | 16-config val-only grid (exploratory; do not repeat test tuning) |
| `scripts/run_subject_cv.py` | 5-fold subject-level CV on the 42 train subjects |

Environment: conda env `dl_env`, Python 3.10.22, torch 2.14.1+cu126,
torchvision 0.29.1+cu126, RTX 4060 Laptop 8 GB. `python -m pytest -q` → 19 passed.

Feature cache: `features/mobilenet_v3_small_16x576/` (415 MB),
`(22553, 16, 576)` float16, aligned with `features/windows.csv`.
A manual recomputation of three windows matched the cache within fp16
precision (max abs diff ≈ 0.06, mean ≈ 0.005), so the cache is not misaligned.

## Main comparison: pre-registered default config, 4 cells × 3 seeds

Command and log:

```bash
python -m src.model.train --cell all --seeds 42 43 44 --owner cai \
  --notes "pre-registered default config" \
  --experiment-log experiments/main/runs.csv \
  --checkpoint-dir experiments/main/checkpoints \
  --log-dir experiments/main/logs
```

Default config: dropout 0.2, hidden 128, lr 1e-3, weight decay 1e-4,
30 epochs, early-stopping patience 5, batch 64.

| Variant | val Macro-F1 | test Macro-F1 | video mean-prob | video majority | test QWK | test F1 95% subject-bootstrap CI |
|---|---|---|---|---|---|---|
| CNN-Frame | 0.371 ± 0.006 | 0.232 ± 0.015 | 0.228 | 0.176 | -0.175 | 0.153–0.299 |
| CNN-Frame-Prior | 0.434 ± 0.008 | 0.253 ± 0.023 | 0.235 | 0.235 | -0.126 | 0.167–0.331 |
| CNN-GRU | 0.388 ± 0.021 | 0.253 ± 0.013 | 0.277 | 0.293 | -0.113 | 0.181–0.327 |
| CNN-GRU-Fusion | 0.439 ± 0.023 | **0.268 ± 0.021** | 0.224 | 0.224 | -0.052 | 0.183–0.348 |
| **RF-Prior (reference)** | 0.433 | **0.453** | — | — | 0.170 (seed 42) | — |
| Majority baseline | — | 0.167 | — | — | 0 | — |

The 3 seeds agree: every CNN cell is clearly below RF-Prior. Video-level
aggregation does not rescue the result, and test QWK stays negative.

## Earlier exploratory results (seed 42, frozen default)

| Variant | val Macro-F1 | test Macro-F1 |
|---|---|---|
| CNN-Frame | 0.364 | 0.228 |
| CNN-Frame-Prior | 0.443 | 0.227 |
| CNN-GRU | 0.408 | 0.268 |
| CNN-GRU-Fusion | 0.419 | 0.247 |

A 16-config val-only grid selected `gru_prior` with dropout 0.5, hidden 64,
lr 1e-4, wd 1e-4 (val 0.503). Its test Macro-F1 was **0.175**, i.e. worse than
the default (`val 0.419 / test 0.247`). This is the clearest evidence that the
9-subject validation is too small for model selection.

A linear probe (StandardScaler + LogisticRegression) on the same cached
features reached val ≈ 0.33–0.34 and test ≈ 0.22–0.24, confirming that the
frozen representation, not just the MLP, is the bottleneck.

## 5-fold subject-level CV (42 train subjects only)

Command and log:

```bash
python scripts/run_subject_cv.py --owner cai
```

80 runs = 5 folds × 8 configs × 2 prior cells; every run used `--skip-test`.
Folds are grouped by subject (9/9/8/8/8 subjects) and never touch the 9 test
subjects.

| Cell | Config | CV val Macro-F1 | vs default | Fold wins | Wilcoxon p | QWK delta |
|---|---|---|---|---|---|---|
| frame_prior | default `base` | 0.369 ± 0.070 | — | — | — | — |
| frame_prior | best `drop5_lr4_wd2` | 0.430 ± 0.055 | +0.061 | 4/5 | 0.125 | +0.135 (5/5) |
| gru_prior | default `base` | 0.407 ± 0.058 | — | — | — | — |
| gru_prior | best `small_lr4_wd2` | 0.436 ± 0.049 | +0.029 | 4/5 | 0.313 | +0.101 (4/5) |

Decision: the CV-best configs are within one fold standard deviation and not
statistically distinguishable (p > 0.1). Per the pre-agreed rule, the **default
configuration remains the main recipe**; the CV-best configs are recorded only
as a sensitivity check. QWK improves consistently for `frame_prior`, so this
can be mentioned in the report, but it does not change the main comparison.

## Decisions and caveats

- **Strict negative result**: frozen ImageNet features are not competitive on
  subject-disjoint UTA-RLDD. RF-Prior remains the strongest baseline.
- **Do not tune on the 9-subject val/test again.** Use the 42-subject CV for
  selection; keep the test split for final evaluation.
- **Test-peeking caveat**: the test split has already been evaluated in
  exploratory runs (4 cells seed 42, the tuned config, and the 12-run main
  comparison). Current test numbers must be labelled exploratory in the
  midterm report. For a clean future number, use a new external dataset
  (NTHU-DDD) or disclose the repeated evaluation explicitly.
- **Bottleneck**: frozen visual features and subject shift, not preprocessing,
  labels, masks, or cache alignment.

## Artifacts (all gitignored)

```
experiments/main/runs.csv              # 12 main runs, window/video/bootstrap
experiments/cv/runs.csv                # 80 CV runs
experiments/cv/summary.csv             # CV mean/std by cell/config
experiments/cv/summary_folds.csv       # per-fold CV rows
experiments/tuning/summary.csv         # exploratory 16-config val-only grid
experiments/runs.csv                   # exploratory seed-42 runs
experiments/{main,cv,tuning}/...        # checkpoints, JSON logs, console logs
features/mobilenet_v3_small_16x576/    # 415 MB feature cache
features/cache_smoke/                  # 198-window smoke cache
```

## Reproduction

```powershell
conda activate dl_env
cd "D:\新建文件夹\one\大三上\深度学习\project"

python -m pytest -q

# Full cache (already built; ~5-15 min on RTX 4060)
python scripts/extract_features.py `
  --output-dir features/mobilenet_v3_small_16x576 `
  --num-workers 4 --batch-windows 32 --frame-batch 256

# Main comparison
python -m src.model.train --cell all --seeds 42 43 44 --owner cai `
  --notes "pre-registered default config" `
  --experiment-log experiments/main/runs.csv `
  --checkpoint-dir experiments/main/checkpoints `
  --log-dir experiments/main/logs

# Subject-level CV (validation only)
python scripts/run_subject_cv.py --owner cai

# Exploratory tuning grid (do not use test for selection)
python scripts/tune_frozen_heads.py --seeds 42 --owner cai
```

## Next session

1. Commit the W11 code and this record to a new branch (suggested
   `w11-model-training`) and push it so the teammate can reproduce.
2. Mao: freeze `splits/within_subject_split.csv` (or an equivalent
   `npz_path,within_split` table) and rerun RF-Prior on it.
3. Attach the within-subject assignment to the cache index (current
   `train.py` reads a split column from `index.csv`; a small merge helper is
   still needed), then run the default 4 cells × 3 seeds with
   `--split-column within_split`.
4. Build the midterm table: RF + 4 cells × both splits + video-level
   Macro-F1 + subject-level bootstrap CI. Report the strict negative result
   and the H3 honest gap.
5. After the midterm, decide whether to implement B: fine-tune the backbone
   with augmentation. If yes, select hyperparameters with subject-level CV
   on the 42 train subjects, never on the test split.

## Open questions

- Which within-subject split exactly (per subject × class 70/15/15, or
  contiguous time blocks)? Mao should freeze and document it before any run.
- Does the report keep the frozen-feature CNN as the primary negative result,
  or is a fine-tuned backbone required for the final comparison?
- How to disclose the repeated test evaluation in the midterm report; a clean
  external validation (NTHU-DDD) is the safest option later.
