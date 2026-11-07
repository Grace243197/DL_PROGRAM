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
& .\scripts\watch_pipeline.ps1
.\.venv\Scripts\python.exe -m src.data.kaggle_pipeline status
.\.venv\Scripts\python.exe -m src.data.rf_baseline
```

The launcher maps the same project directory to `Z:` so MediaPipe can load its
bundled model despite the Chinese parent path. The alias is recreated when needed;
if `Z:` is occupied by another drive, choose a free letter in the launcher.

If a video pauses for face-detection failures caused by a small face, retain its
original video, archive that video's failed status marker and processed directory,
then reprocess only its key with `run --only-key KEY --face-detection-fallback`.
This opt-in pass uses full-range face detection to locate a small face before
running Face Mesh on the face region. Record its use in the video summary and
check the resulting failure rate before resuming the full queue.

If review shows genuine face occlusion rather than a detector defect, keep that
video's source and quality-hold marker. `start_pipeline.ps1 -Mode run
-ContinuePastQualityHolds` skips reviewed held videos while processing later ones.
It does not run the RF baseline; unresolved holds must be decided and documented
before the final feature consolidation and evaluation.

The pilot processes the first subject's three classes. Review its frame crops,
failure rates and window counts before `run`. The full command resumes from
per-video status markers. Each video is downloaded into `data/staging`, extracted if
Kaggle wraps it in ZIP,
processed into 5-second non-overlapping windows with 16 sampled 160x160 face frames,
and verified. It retains the processed `.npz` windows and feature tables. Only after
verification does it delete its own downloaded ZIP and extracted source video. If
the face detection failure rate exceeds 10%, the source is retained and the queue
pauses for review. MOV rotation metadata is applied before face detection.

`features/windows.csv` is consolidated from completed videos. A final RF run is
blocked until every manifest video has a completion marker; `--allow-partial`
produces an explicitly labelled pilot result only when all three splits and classes
are represented.

The sampled-frame ECR, blink and yawn event counts are approximations. Sampling 16
frames over 5 seconds cannot resolve all short blinks. Head pose uses approximate
camera intrinsics and should not be treated as calibrated physical angle. Video
windowing uses the nominal FPS reported by OpenCV; variable-FPS files need an
additional timestamp audit before final reporting.

The full-run launcher also starts `scripts/finish_w9.ps1`. It waits for the download
process, retries an interrupted queue up to three times, and runs RF-Prior only
after all manifest videos are complete. The launcher blocks duplicate live full
runs. A machine shutdown stops both background processes; start the full run
again after reboot to resume.

`watch_pipeline.ps1` refreshes every 10 seconds and shows completed videos, the
current partial-download size, or the number of processed windows. Press Ctrl+C
to stop watching; use `-Once` for a single snapshot. The download size is received
ZIP bytes, so it is not a reliable per-video completion percentage.
