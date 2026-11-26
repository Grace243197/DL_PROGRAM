# Within-subject split (H3 protocol)

Frozen: 2026-11-26. Owner: 毛禹童. Status: **frozen**, do not regenerate unless the
protocol is intentionally changed.

## Why this file exists

The project's primary evaluation is the **subject-disjoint** 42/9/9 split in
[`subject_split.json`](subject_split.json). The proposal also requires every
variant to be evaluated **within-subject**, so that H3 (the honest gap between
within-subject and subject-disjoint numbers) can be reported.

This directory now contains both splits. They are independent:

| File | Protocol | Use |
|---|---|---|
| `subject_split.json` | 42/9/9 subjects, no subject in two sets | primary result |
| `within_subject_split.csv` | every subject appears in train/val/test; windows split 70/15/15 | H3 contrast only |

The original `split` column in `features/windows.csv` is **not modified**.
`within_subject_split.csv` adds a separate `within_split` column.

## Frozen protocol

- **Source of truth**: `features/windows.csv` (22,553 accepted windows). Do not
  glob `data/windows/**/*.npz`; 76 rejected windows are not part of this split.
- **Grouping**: 180 `(subject_id, label)` groups.
- **Ordering**: within each group, windows are sorted by `source_file` then
  `window_index`.
- **Assignment**: shuffle the sorted windows with seed **20261105**, then assign
  70% / 15% / 15% to train / val / test. Each group keeps all three classes by
  construction.
- **Windows are non-overlapping 5-second windows.** A random window split
  therefore does not leak identical frames; residual temporal correlation
  between adjacent windows remains a disclosed caveat.
- **Original split untouched**: `subject_split` in the output is a copy of the
  original subject-disjoint `split` column, for audit only.

## Counts

| Split | Windows | Share | Alert | Low-vigilant | Drowsy | Subjects |
|---|---:|---:|---:|---:|---:|---:|
| train | 15,797 | 70.04% | 5,227 | 5,269 | 5,301 | 60 |
| val | 3,378 | 14.98% | 1,116 | 1,128 | 1,134 | 60 |
| test | 3,378 | 14.98% | 1,116 | 1,128 | 1,134 | 60 |
| **total** | **22,553** | 100% | 7,459 | 7,525 | 7,569 | 60 |

## Files

| File | SHA-256 |
|---|---|
| `splits/within_subject_split.csv` | `43bcc072d8db9aff1b27d66968a6184d2b763f9d62dcd0cdb1677228b8c96a45` |
| `splits/within_subject_split.meta.json` | see the file; it records the source manifest hash and counts |

CSV columns:

```text
npz_path, subject_id, source_file, video_key, label, class_name,
window_index, subject_split, within_split
```

`npz_path` is the merge key. `subject_split` is the original subject-disjoint
assignment; `within_split` is the new train/val/test assignment.

## How to use

### RF-Prior

```powershell
& .\.venv\Scripts\python.exe -m src.data.rf_baseline `
  --split-column within_split `
  --split-file splits/within_subject_split.csv `
  --allow-subject-overlap `
  --tag within
```

`--allow-subject-overlap` is required because every subject intentionally
appears in all three sets. Do **not** apply the subject-disjoint leakage check
to this split.

### Deep models (蔡)

The cache index (`features/mobilenet_v3_small_16x576/index.csv`) already
contains `npz_path` and `manifest_row`. Merge `within_split` on `npz_path`
(or `manifest_row`), write a **new** index file, and point the cache dataset at
it; then run the four cells with `--split-column within_split`. Do not overwrite
the original `index.csv`.

## Frozen result: RF-Prior

Run: `results/rf_prior_full_within_20261126T071228Z.json` (3 seeds 42/43/44).

| Protocol | val Macro-F1 | test Macro-F1 | test QWK |
|---|---:|---:|---:|
| Subject-disjoint (W9) | 0.433 | 0.453 | 0.170 |
| **Within-subject (this split)** | **0.811 ± 0.003** | **0.816 ± 0.004** | **0.749** |

Within-subject test per-class recall (best-val seed 43): alert 0.848,
low-vigilant 0.790, drowsy 0.819. Detection failure rate on the within test set:
0.065% (well below the 10% stop line).

The gap is **+0.363 Macro-F1** (0.453 → 0.816). This is the H3 evidence: the
same RF-Prior features look much stronger when the same subjects appear in
train and test. The two test sets are different windows (3,323 unseen subjects
vs 3,378 mixed-subject windows), so this is a protocol-level contrast, not a
paired comparison; report it that way.

## Regeneration and freeze semantics

```powershell
& .\.venv\Scripts\python.exe -m src.data.within_split
```

- Same config and seed → identical file. The script refuses to overwrite a
  different existing file.
- `--force` is only for an intentional protocol change; if used, update this
  README, the hash, and the W12 handoff.
- Unit tests: `python -m pytest tests/test_within_split.py -q` (6 tests).

## Caveats to carry into the report

1. The within test set was **not** used for model selection. The RF
   hyperparameters come from W9 and were not tuned on this split. Keep it that
   way.
2. The subject-disjoint test set has already been evaluated repeatedly and is
   labelled *exploratory* (see W11 handoff). The within test set is fresh; still
   evaluate it once per final configuration.
3. Adjacent non-overlapping windows are temporally correlated. Do not describe
   this split as "frame-level random"; it is window-level and non-overlapping.
4. This split is for the H3 contrast; the primary claim remains the
   subject-disjoint result.
