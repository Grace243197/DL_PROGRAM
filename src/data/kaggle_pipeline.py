from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd
import requests

from .common import ROOT, load_config, read_json, write_json_atomic
from .facial_features import FaceExtractor, summarize_priors
from .split import freeze_split


VIDEO_NAME = re.compile(r"^(0|5|10)(?:_\d+)?\.(?:mov|mp4|m4v)$", re.IGNORECASE)


def token_from_environment() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    secret_file = ROOT / ".secrets" / "kaggle_token.txt"
    if not token and secret_file.exists():
        token = secret_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(
            "Kaggle token missing. Set KAGGLE_API_TOKEN or place it in "
            ".secrets/kaggle_token.txt (this path is gitignored)."
        )
    return token


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token_from_environment()}",
                            "User-Agent": "uta-rldd-w9-pipeline/1.0"})
    return session


def _api_json(session: requests.Session, url: str) -> dict:
    for attempt in range(1, 9):
        try:
            response = session.get(url, timeout=(20, 90))
            if response.status_code in (401, 403, 404):
                code = response.status_code
                response.close()
                raise RuntimeError(f"Kaggle metadata HTTP {code}; check token and dataset handle")
            if response.status_code == 429 or 500 <= response.status_code < 600:
                delay = _backoff(attempt, response.headers.get("Retry-After"))
                response.close()
                print(f"Metadata HTTP error; retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            if attempt == 8:
                raise RuntimeError("Kaggle metadata unavailable after 8 attempts") from error
            delay = _backoff(attempt)
            print(f"Metadata request interrupted; retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("Kaggle metadata unavailable after 8 attempts")


def parse_entry(file: dict) -> dict | None:
    name = file["name"]
    parts = name.split("/")
    if len(parts) < 2 or not VIDEO_NAME.fullmatch(parts[-1]):
        return None
    subject = parts[-2]
    if not re.fullmatch(r"\d{2}", subject):
        return None
    label = VIDEO_NAME.fullmatch(parts[-1]).group(1)
    key = f"{subject}_{label}_{hashlib.sha256(name.encode()).hexdigest()[:12]}"
    return {"name": name, "subject_id": subject, "label": label,
            "bytes": int(file.get("totalBytes") or 0), "key": key}


def fetch_manifest(session: requests.Session, config: dict) -> dict:
    dataset = config["dataset"]
    base = f"https://www.kaggle.com/api/v1/datasets"
    identity = f"{dataset['owner']}/{dataset['slug']}"
    view = _api_json(session, f"{base}/view/{identity}")
    files = []
    page_token = None
    while True:
        url = f"{base}/list/{identity}?pageSize=200"
        if page_token:
            url += "&pageToken=" + quote(page_token, safe="")
        page = _api_json(session, url)
        files.extend(page.get("datasetFiles", []))
        page_token = page.get("nextPageToken")
        if not page_token:
            break
    entries = [item for file in files if (item := parse_entry(file)) is not None]
    entries.sort(key=lambda item: (item["subject_id"], int(item["label"]), item["name"]))
    expected_subjects = {f"{index:02d}" for index in range(1, dataset["expected_subjects"] + 1)}
    observed_subjects = {item["subject_id"] for item in entries}
    if observed_subjects != expected_subjects:
        raise RuntimeError(f"Unexpected subject set: missing={sorted(expected_subjects - observed_subjects)}, "
                           f"extra={sorted(observed_subjects - expected_subjects)}")
    coverage = {(item["subject_id"], item["label"]) for item in entries}
    missing = sorted((subject, label) for subject in expected_subjects
                     for label in dataset["labels"] if (subject, label) not in coverage)
    if missing:
        raise RuntimeError(f"Dataset lacks required subject/class pairs: {missing}")
    if len({item["key"] for item in entries}) != len(entries):
        raise RuntimeError("Duplicate manifest keys")
    return {"dataset": identity, "version": view["currentVersionNumber"],
            "total_bytes": view["totalBytes"], "files": entries}


def freeze_manifest(session: requests.Session, config: dict) -> dict:
    target = ROOT / "splits" / "source_manifest.json"
    if target.exists():
        manifest = read_json(target)
        if manifest["dataset"] != f"{config['dataset']['owner']}/{config['dataset']['slug']}":
            raise RuntimeError("Frozen manifest belongs to another dataset")
        return manifest
    manifest = fetch_manifest(session, config)
    write_json_atomic(target, manifest)
    return manifest


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(900.0, max(1.0, float(retry_after)))
        except ValueError:
            pass
    return min(900.0, 5.0 * (2 ** (attempt - 1)))


def download_one(session: requests.Session, config: dict, entry: dict, version: int) -> Path:
    stage = ROOT / "data" / "staging"
    stage.mkdir(parents=True, exist_ok=True)
    archive = stage / f"{entry['key']}.zip"
    partial = stage / f"{entry['key']}.zip.part"
    if archive.exists():
        return archive
    if partial.exists() and zipfile.is_zipfile(partial):
        with zipfile.ZipFile(partial) as package:
            if package.testzip() is None:
                os.replace(partial, archive)
                return archive
    dataset = config["dataset"]
    url = (f"https://www.kaggle.com/api/v1/datasets/download/"
           f"{dataset['owner']}/{dataset['slug']}?datasetVersionNumber={version}"
           f"&fileName={quote(entry['name'], safe='')}")
    settings = config["download"]
    for attempt in range(1, settings["max_attempts"] + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with session.get(url, headers=headers, stream=True, allow_redirects=True,
                             timeout=(settings["connect_timeout_seconds"],
                                      settings["read_timeout_seconds"])) as response:
                if response.status_code in (401, 403, 404):
                    raise RuntimeError(f"Kaggle download HTTP {response.status_code} for {entry['name']}")
                if response.status_code == 416 and existing:
                    partial.unlink()
                    continue
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    delay = _backoff(attempt, response.headers.get("Retry-After"))
                    print(f"HTTP {response.status_code}; retrying in {delay:.0f}s: {entry['key']}", flush=True)
                    time.sleep(delay)
                    continue
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP {response.status_code} for {entry['name']}")
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {existing}-"):
                        raise RuntimeError(f"Invalid resume offset for {entry['name']}: {content_range}")
                mode = "ab" if response.status_code == 206 and existing else "wb"
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=settings["chunk_bytes"]):
                        if chunk:
                            output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            os.replace(partial, archive)
            return archive
        except (requests.RequestException, OSError) as error:
            if attempt == settings["max_attempts"]:
                raise RuntimeError(f"Download failed after {attempt} attempts: {entry['name']}") from error
            delay = _backoff(attempt)
            print(f"Download {type(error).__name__}; retrying in {delay:.0f}s: {entry['key']}", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"Download attempts exhausted: {entry['name']}")


def unpack_one(archive: Path, entry: dict) -> Path:
    output = archive.with_name(entry["key"] + Path(entry["name"]).suffix.lower())
    if output.exists():
        return output
    temporary = output.with_name(output.name + ".part")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as package:
            members = [member for member in package.infolist()
                       if not member.is_dir() and Path(member.filename).name.lower()
                       == Path(entry["name"]).name.lower()]
            if len(members) != 1:
                raise RuntimeError(f"Expected one video in {archive}, found {len(members)}")
            with package.open(members[0]) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    else:
        with archive.open("rb") as source:
            header = source.read(12)
        if header.startswith(b"PK") or b"ftyp" not in header:
            raise RuntimeError(f"Incomplete or unrecognized Kaggle archive: {archive}")
        if entry["bytes"] and archive.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Downloaded video size differs from Kaggle manifest: {entry['name']}")
        os.replace(archive, output)
        return output
    os.replace(temporary, output)
    return output


def prepare_video(session: requests.Session, config: dict, entry: dict, version: int) -> tuple[Path, Path]:
    stage = ROOT / "data" / "staging"
    archive = stage / f"{entry['key']}.zip"
    output = stage / (entry["key"] + Path(entry["name"]).suffix.lower())
    if output.exists():
        return archive, output
    for attempt in range(2):
        archive = download_one(session, config, entry, version)
        try:
            return archive, unpack_one(archive, entry)
        except (RuntimeError, zipfile.BadZipFile, OSError) as error:
            archive.unlink(missing_ok=True)
            archive.with_name(archive.name + ".part").unlink(missing_ok=True)
            if attempt:
                raise RuntimeError(f"Cannot unpack Kaggle video after a fresh download: {entry['name']}") from error
            print(f"Archive invalid; downloading again: {entry['key']}", flush=True)
    raise RuntimeError(f"Cannot prepare {entry['name']}")


def _split_for(subject: str, split: dict) -> str:
    matches = [name for name in ("train", "val", "test") if subject in split[name]]
    if len(matches) != 1:
        raise RuntimeError(f"Subject {subject} is absent or duplicated in frozen split")
    return matches[0]


def _read_target(cap: cv2.VideoCapture, position: int, current: int) -> tuple[np.ndarray, int]:
    while current < position:
        if not cap.grab():
            raise RuntimeError(f"Video ended before frame {position}")
        current += 1
    ok, frame = cap.retrieve()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot decode frame {position}")
    return frame, current


def process_video(video: Path, entry: dict, split: dict, config: dict) -> dict:
    settings = config["preprocessing"]
    set_name = _split_for(entry["subject_id"], split)
    relative = Path(set_name) / entry["subject_id"] / entry["label"] / entry["key"]
    final_dir = ROOT / "data" / "windows" / relative
    work_dir = ROOT / "data" / "work" / entry["key"]
    if final_dir.exists():
        return validate_processed(final_dir, entry, set_name)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}; the source is retained")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_frames = int(round(fps * settings["window_seconds"]))
    if not np.isfinite(fps) or fps <= 0 or frames < window_frames or window_frames < settings["frames_per_window"]:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: fps={fps}, frames={frames}, file={video}")
    rows = []
    frame_index = -1
    try:
        with FaceExtractor(settings) as extractor:
            for window_index in range(frames // window_frames):
                start = window_index * window_frames
                targets = np.linspace(start, start + window_frames - 1,
                                      settings["frames_per_window"], dtype=int)
                crops, priors, valid = [], [], []
                for target in targets:
                    frame, frame_index = _read_target(cap, int(target), frame_index)
                    result = extractor.extract(frame)
                    crops.append(result.crop)
                    priors.append(result.prior)
                    valid.append(result.detected)
                crop_array = np.stack(crops).astype(np.uint8)
                prior_array = np.stack(priors).astype(np.float32)
                valid_array = np.asarray(valid, dtype=bool)
                npz_name = f"w{window_index:06d}.npz"
                np.savez_compressed(work_dir / npz_name, frames=crop_array,
                                    priors=prior_array, valid_mask=valid_array)
                row = {
                    "source_file": entry["name"], "video_key": entry["key"],
                    "subject_id": entry["subject_id"], "split": set_name,
                    "label": int(entry["label"]),
                    "class_name": config["dataset"]["labels"][entry["label"]],
                    "window_index": window_index,
                    "start_s": round(start / fps, 4),
                    "end_s": round((start + window_frames) / fps, 4),
                    "npz_path": (Path("data") / "windows" / relative / npz_name).as_posix(),
                    **summarize_priors(prior_array, valid_array, settings),
                }
                rows.append(row)
    finally:
        cap.release()
    if not rows:
        raise RuntimeError(f"No complete windows produced for {video}")
    pd.DataFrame(rows).to_csv(work_dir / "features.csv", index=False)
    summary = {
        "source_file": entry["name"], "video_key": entry["key"],
        "split": set_name, "subject_id": entry["subject_id"],
        "label": int(entry["label"]), "fps": fps,
        "source_frames": frames, "window_count": len(rows),
        "face_size": settings["face_size"],
        "sampled_frames": len(rows) * settings["frames_per_window"],
        "detection_failure_rate": float(np.mean([row["detection_failure_rate"] for row in rows])),
        "output_dir": str(final_dir.relative_to(ROOT)).replace("\\", "/"),
    }
    write_json_atomic(work_dir / "summary.json", summary)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(work_dir, final_dir)
    return validate_processed(final_dir, entry, set_name)


def validate_processed(directory: Path, entry: dict, set_name: str) -> dict:
    summary = read_json(directory / "summary.json")
    if summary["video_key"] != entry["key"] or summary["split"] != set_name:
        raise RuntimeError(f"Output metadata mismatch: {directory}")
    count = summary["window_count"]
    paths = sorted(directory.glob("w*.npz"))
    if count < 1 or len(paths) != count:
        raise RuntimeError(f"Output window count mismatch: {directory}")
    table = pd.read_csv(directory / "features.csv")
    if len(table) != count or table["video_key"].nunique() != 1:
        raise RuntimeError(f"Output feature count mismatch: {directory}")
    for path in (paths[0], paths[-1]):
        with np.load(path) as bundle:
            if bundle["frames"].shape[0] != summary["sampled_frames"] // count:
                raise RuntimeError(f"Output frame count mismatch: {path}")
            if bundle["frames"].shape[1:3] != (summary["face_size"], summary["face_size"]):
                raise RuntimeError(f"Output face size mismatch: {path}")
    return summary


def consolidate(manifest: dict) -> dict:
    status_dir = ROOT / "data" / "status"
    rows = []
    counts = Counter()
    failure_sum = Counter()
    for entry in manifest["files"]:
        status_file = status_dir / f"{entry['key']}.json"
        if not status_file.exists():
            continue
        status = read_json(status_file)
        path = ROOT / status["output_dir"] / "features.csv"
        if not path.exists():
            raise RuntimeError(f"Missing feature table for completed video: {entry['name']}")
        frame = pd.read_csv(path)
        rows.append(frame)
        counts[(status["split"], status["label"])] += len(frame)
        failure_sum[status["split"]] += float(frame["detection_failure_rate"].sum())
    if rows:
        output = ROOT / "features" / "windows.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".part")
        pd.concat(rows, ignore_index=True).to_csv(temporary, index=False)
        os.replace(temporary, output)
    windows_by_split = Counter()
    for (name, _), count in counts.items():
        windows_by_split[name] += count
    return {"videos_done": len(rows), "videos_total": len(manifest["files"]),
            "windows_by_split_class": {f"{s}/{label}": count for (s, label), count in sorted(counts.items())},
            "failure_rate_by_split": {name: failure_sum[name] / count
                                      for name, count in windows_by_split.items()}}


def run(config: dict, limit_files: int | None = None) -> dict:
    split = freeze_split()
    session = make_session()
    manifest = freeze_manifest(session, config)
    status_dir = ROOT / "data" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    selected = manifest["files"][:limit_files] if limit_files else manifest["files"]
    for index, entry in enumerate(selected, 1):
        marker = status_dir / f"{entry['key']}.json"
        if marker.exists():
            status = read_json(marker)
            validate_processed(ROOT / status["output_dir"], entry,
                               _split_for(entry["subject_id"], split))
            if not status.get("source_retained_for_review", False):
                archive = ROOT / "data" / "staging" / f"{entry['key']}.zip"
                source = archive.with_name(entry["key"] + Path(entry["name"]).suffix.lower())
                source.unlink(missing_ok=True)
                archive.unlink(missing_ok=True)
            print(f"[{index}/{len(selected)}] already complete: {entry['name']}", flush=True)
            continue
        if shutil.disk_usage(ROOT).free < 20 * 1024 ** 3:
            raise RuntimeError("Less than 20 GiB free; pausing before next video")
        set_name = _split_for(entry["subject_id"], split)
        final_dir = (ROOT / "data" / "windows" / set_name / entry["subject_id"]
                     / entry["label"] / entry["key"])
        archive = ROOT / "data" / "staging" / f"{entry['key']}.zip"
        video = archive.with_name(entry["key"] + Path(entry["name"]).suffix.lower())
        if final_dir.exists():
            summary = validate_processed(final_dir, entry, set_name)
        else:
            print(f"[{index}/{len(selected)}] downloading {entry['name']}", flush=True)
            archive, video = prepare_video(session, config, entry, manifest["version"])
            summary = process_video(video, entry, split, config)
        keep_source = summary["detection_failure_rate"] > config["preprocessing"]["max_failure_rate_for_cleanup"]
        summary["source_retained_for_review"] = keep_source
        write_json_atomic(marker, summary)
        if not keep_source:
            video.unlink(missing_ok=True)
            archive.unlink(missing_ok=True)
        print(f"  windows={summary['window_count']} failure={summary['detection_failure_rate']:.1%}"
              f" retained={keep_source}", flush=True)
    result = consolidate(manifest)
    print(result, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental UTA-RLDD W9 pipeline")
    parser.add_argument("command", choices=("manifest", "pilot", "run", "status"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit-files", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "manifest":
        manifest = freeze_manifest(make_session(), config)
        print(f"{manifest['dataset']} version={manifest['version']} videos={len(manifest['files'])}")
    elif args.command == "pilot":
        run(config, args.limit_files)
    elif args.command == "run":
        run(config)
    else:
        manifest = read_json(ROOT / "splits" / "source_manifest.json")
        print(consolidate(manifest))


if __name__ == "__main__":
    main()
