# W9 UTA-RLDD incremental pipeline

The frozen subject split is `splits/subject_split.json` (42 train / 9 validation / 9 test,
seed 20261105). The source manifest is frozen at `splits/source_manifest.json` after the
first successful Kaggle metadata request. Labels are `0=alert`, `5=low_vigilant`, and
`10=drowsy`. All windows from one subject stay in one split.

## Setup

In PowerShell, from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
```

Put a Kaggle API token alone on one line in `.secrets/kaggle_token.txt`, or set
`KAGGLE_API_TOKEN`. The `.secrets` directory is ignored by Git. Never commit or paste
the token into code, logs, or a report.

```powershell
.\.venv\Scripts\python.exe -m src.data.split
.\.venv\Scripts\python.exe -m src.data.kaggle_pipeline manifest
& .\scripts\start_pipeline.ps1 -Mode pilot -LimitFiles 3
& .\scripts\start_pipeline.ps1 -Mode run
.\.venv\Scripts\python.exe -m src.data.kaggle_pipeline status
.\.venv\Scripts\python.exe -m src.data.rf_baseline
```

The launcher maps the same project directory to `Z:` so MediaPipe can load its
bundled model despite the Chinese parent path. The alias is recreated when needed;
if `Z:` is occupied by another drive, choose a free letter in the launcher.

The pilot processes the first subject's three classes. Review its frame crops,
failure rates and window counts before `run`. The full command resumes from
per-video status markers. Each video is downloaded into `data/staging`, extracted if
Kaggle wraps it in ZIP,
processed into 5-second non-overlapping windows with 16 sampled 160x160 face frames,
and verified. It retains the processed `.npz` windows and feature tables. Only after
verification does it delete its own downloaded ZIP and extracted source video. If
the face detection failure rate exceeds 10%, the source is retained for review.

`features/windows.csv` is consolidated from completed videos. A final RF run is
blocked until every manifest video has a completion marker; `--allow-partial`
produces an explicitly labelled pilot result only when all three splits and classes
are represented.

The sampled-frame ECR, blink and yawn event counts are approximations. Sampling 16
frames over 5 seconds cannot resolve all short blinks. Head pose uses approximate
camera intrinsics and should not be treated as calibrated physical angle. Video
windowing uses the nominal FPS reported by OpenCV; variable-FPS files need an
additional timestamp audit before final reporting.

