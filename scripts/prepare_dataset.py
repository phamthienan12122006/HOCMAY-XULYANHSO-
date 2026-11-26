#!/usr/bin/env python3
"""Audit raw knee X-ray images and generate manifest.csv and split.csv.

Reads untouched raw images from `data/raw/KL_0/` .. `data/raw/KL_4/` and source
provenance from `data/metadata/source_index.csv`, audits every image for
readability, dimensions, channels, SHA-256 hash, pixel-content hash, and label
integrity, then outputs:
    - `data/metadata/manifest.csv`
    - `data/metadata/split.csv`
    - `data/metadata/audit_details.json`

Never modifies original images and never drops low-quality images automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

# Keep strict truncation detection enabled so truncated/corrupt files are caught!
ImageFile.LOAD_TRUNCATED_IMAGES = False

VALID_GRADES = {0, 1, 2, 3, 4}
SPLIT_MAP = {
    "train": "train",
    "val": "validation",
    "test": "test",
}
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
FILENAME_PATTERN = re.compile(r"^(\d+)([LR])\.png$")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_image(path: Path) -> Tuple[dict, dict]:
    """Strictly verify and decode an image file without modifying it."""
    meta = {
        "width": None,
        "height": None,
        "channels": None,
        "image_format": None,
        "status": "ok",
    }
    extra = {
        "color_mode": None,
        "pixel_sha256": None,
        "mean_intensity": None,
        "std_intensity": None,
        "min_intensity": None,
        "max_intensity": None,
        "error_detail": None,
    }

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            bands = img.getbands()
            meta["width"] = int(width)
            meta["height"] = int(height)
            meta["channels"] = int(len(bands))
            meta["image_format"] = str(img.format)

            extra["color_mode"] = str(img.mode)
            pixel_bytes = img.tobytes()
            pix_hash = hashlib.sha256()
            pix_hash.update(f"{img.mode}:{width}x{height}:".encode("ascii"))
            pix_hash.update(pixel_bytes)
            extra["pixel_sha256"] = pix_hash.hexdigest()

            arr = np.asarray(img, dtype=np.float32)
            extra["mean_intensity"] = float(arr.mean())
            extra["std_intensity"] = float(arr.std())
            extra["min_intensity"] = float(arr.min())
            extra["max_intensity"] = float(arr.max())
    except Exception as exc:
        meta["status"] = "corrupted"
        extra["error_detail"] = f"{type(exc).__name__}: {exc}"

    return meta, extra


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit raw images and generate manifest.csv and split.csv."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the repository root.",
    )
    args = parser.parse_args()

    project_root: Path = args.project_root.resolve()
    raw_dir = project_root / "data" / "raw"
    meta_dir = project_root / "data" / "metadata"
    source_index_path = meta_dir / "source_index.csv"

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. Run scripts/download_dataset.py first."
        )
    if not source_index_path.is_file():
        raise FileNotFoundError(
            f"Source index not found: {source_index_path}. Run scripts/download_dataset.py first."
        )

    # Load source_index.csv for traceable official source_split mapping
    source_lookup: Dict[str, dict] = {}
    with source_index_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            source_lookup[row["relative_path"]] = row

    # Discover every file under data/raw recursively so no unlabeled/stray file goes unnoticed
    discovered_files = sorted(p for p in raw_dir.rglob("*") if p.is_file())
    print(f"[INFO] Auditing {len(discovered_files):,} files under {raw_dir}...")

    manifest_rows: List[dict] = []
    unlabeled_or_unexpected: List[str] = []
    corrupted_files: List[dict] = []
    low_contrast_warnings: List[dict] = []
    pixel_hash_map: Dict[str, List[str]] = defaultdict(list)
    file_hash_map: Dict[str, List[str]] = defaultdict(list)
    color_modes: Counter = Counter()
    resolutions: Counter = Counter()
    formats: Counter = Counter()
    non_standard_filenames: List[str] = []
    missing_from_source_index: List[str] = []

    for idx, path in enumerate(discovered_files, start=1):
        rel_path = path.relative_to(project_root).as_posix()
        parent_name = path.parent.name

        # Determine KL grade strictly from KL_<0..4> directory
        kl_match = re.fullmatch(r"KL_([0-4])", parent_name)
        if not kl_match:
            unlabeled_or_unexpected.append(rel_path)
            continue
        kl_grade = int(kl_match.group(1))

        src_row = source_lookup.get(rel_path)
        if src_row is None:
            missing_from_source_index.append(rel_path)
            source_split = ""
        else:
            source_split = src_row["source_split"]
            if int(src_row["kl_grade"]) != kl_grade:
                raise RuntimeError(
                    f"Label mismatch for {rel_path}: folder KL={kl_grade} vs source_index KL={src_row['kl_grade']}"
                )

        fname = path.name
        image_id = path.stem

        # Extract verified source patient_id & knee_side ONLY when matching author's exact naming pattern
        fn_match = FILENAME_PATTERN.match(fname)
        if fn_match:
            patient_id = fn_match.group(1)
            knee_side = fn_match.group(2)
        else:
            patient_id = ""
            knee_side = ""
            non_standard_filenames.append(rel_path)

        fsize = path.stat().st_size
        sha256_hex = file_sha256(path)
        img_meta, extra = inspect_image(path)

        status = img_meta["status"]
        if kl_grade not in VALID_GRADES:
            status = "invalid_label"

        if status != "ok":
            corrupted_files.append(
                {
                    "relative_path": rel_path,
                    "status": status,
                    "error_detail": extra["error_detail"],
                }
            )
        else:
            color_modes[extra["color_mode"]] += 1
            resolutions[f"{img_meta['width']}x{img_meta['height']}"] += 1
            formats[img_meta["image_format"]] += 1
            file_hash_map[sha256_hex].append(rel_path)
            pixel_hash_map[extra["pixel_sha256"]].append(rel_path)

            # Record extreme low-contrast / near-blank images for reporting ONLY (never auto-drop)
            if extra["std_intensity"] is not None and extra["std_intensity"] < 15.0:
                low_contrast_warnings.append(
                    {
                        "image_id": image_id,
                        "relative_path": rel_path,
                        "kl_grade": kl_grade,
                        "mean_intensity": round(extra["mean_intensity"], 2),
                        "std_intensity": round(extra["std_intensity"], 2),
                    }
                )

        manifest_rows.append(
            {
                "image_id": image_id,
                "file_name": fname,
                "relative_path": rel_path,
                "kl_grade": kl_grade,
                "width": img_meta["width"],
                "height": img_meta["height"],
                "channels": img_meta["channels"],
                "image_format": img_meta["image_format"],
                "file_size": fsize,
                "sha256": sha256_hex,
                "status": status,
                "patient_id": patient_id,
                "knee_side": knee_side,
                "source_split": source_split,
            }
        )

        if idx % 2000 == 0 or idx == len(discovered_files):
            print(f"       Audited {idx:,} / {len(discovered_files):,} files...", flush=True)

    # Sort manifest deterministically
    manifest_rows.sort(key=lambda r: (r["kl_grade"], r["source_split"], r["image_id"]))
    manifest_df = pd.DataFrame(manifest_rows)

    # Sanity check unique image_id
    if manifest_df["image_id"].duplicated().any():
        dups = manifest_df[manifest_df["image_id"].duplicated(keep=False)]["image_id"].tolist()
        raise RuntimeError(f"Duplicate image_id detected: {dups[:10]}")

    manifest_path = meta_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"[OK] Wrote {manifest_path} ({len(manifest_df):,} rows)")

    # Build split.csv for all valid images (status == 'ok')
    valid_df = manifest_df[manifest_df["status"] == "ok"].copy()
    if valid_df["source_split"].isin(SPLIT_MAP.keys()).all():
        split_strategy = "official_mendeley_patient_grouped_split"
        valid_df["split"] = valid_df["source_split"].map(SPLIT_MAP)
    else:
        raise RuntimeError(
            "Not all valid images have a recognized official source_split (train/val/test)."
        )

    # Verify zero patient leakage across train / validation / test
    if (valid_df["patient_id"] != "").all():
        split_patients = {
            sp: set(grp["patient_id"]) for sp, grp in valid_df.groupby("split")
        }
        overlaps = {
            "train_vs_validation": sorted(
                split_patients.get("train", set()) & split_patients.get("validation", set())
            ),
            "train_vs_test": sorted(
                split_patients.get("train", set()) & split_patients.get("test", set())
            ),
            "validation_vs_test": sorted(
                split_patients.get("validation", set()) & split_patients.get("test", set())
            ),
        }
        if any(overlaps.values()):
            raise RuntimeError(
                f"Patient leakage detected in official split: {overlaps}"
            )
    else:
        overlaps = {}

    split_df = valid_df[["image_id", "split", "kl_grade", "patient_id"]].copy()
    split_df["_order"] = split_df["split"].map(SPLIT_ORDER)
    split_df = split_df.sort_values(["_order", "kl_grade", "image_id"]).drop(columns=["_order"])

    split_path = meta_dir / "split.csv"
    split_df.to_csv(split_path, index=False, encoding="utf-8")
    print(f"[OK] Wrote {split_path} ({len(split_df):,} rows)")

    # Duplicate checks (file SHA-256 and decoded pixel SHA-256)
    exact_file_dups = {h: paths for h, paths in file_hash_map.items() if len(paths) > 1}
    exact_pixel_dups = {h: paths for h, paths in pixel_hash_map.items() if len(paths) > 1}
    # Pixel duplicates that are NOT file duplicates (same content, different metadata/encoding)
    pixel_only_dups = {
        h: paths
        for h, paths in exact_pixel_dups.items()
        if len(set(manifest_df.loc[manifest_df["relative_path"].isin(paths), "sha256"])) > 1
    }

    audit_details = {
        "total_discovered_files": len(discovered_files),
        "total_manifest_rows": len(manifest_df),
        "total_valid_images": len(valid_df),
        "total_corrupted_images": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "unlabeled_or_unexpected_files": unlabeled_or_unexpected,
        "missing_from_source_index": missing_from_source_index,
        "non_standard_filenames": non_standard_filenames,
        "exact_sha256_duplicate_groups": len(exact_file_dups),
        "exact_sha256_duplicates": exact_file_dups,
        "exact_pixel_duplicate_groups": len(exact_pixel_dups),
        "exact_pixel_duplicates": exact_pixel_dups,
        "pixel_same_but_file_diff_groups": len(pixel_only_dups),
        "color_modes": dict(color_modes),
        "resolutions": dict(resolutions),
        "image_formats": dict(formats),
        "low_contrast_warnings_count": len(low_contrast_warnings),
        "low_contrast_warnings_sample": low_contrast_warnings[:20],
        "split_strategy": split_strategy,
        "random_seed_policy": 42,
        "patient_overlap_counts": {k: len(v) for k, v in overlaps.items()},
        "unique_patients_total": int(valid_df["patient_id"].nunique()),
        "unique_patients_by_split": {
            sp: int(grp["patient_id"].nunique()) for sp, grp in valid_df.groupby("split")
        },
        "knees_per_patient_distribution": {
            str(k): int(v)
            for k, v in Counter(valid_df.groupby("patient_id").size()).items()
        },
    }

    audit_path = meta_dir / "audit_details.json"
    audit_path.write_text(json.dumps(audit_details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
