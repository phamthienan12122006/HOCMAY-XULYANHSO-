#!/usr/bin/env python3
"""Download and organize the official Mendeley Knee Osteoarthritis KL Grading Dataset.

Primary source:
    Pingjun Chen (2018), "Knee Osteoarthritis Severity Grading Dataset",
    Mendeley Data, V1, DOI: 10.17632/56rmx5bjcr.1
    URL: https://data.mendeley.com/datasets/56rmx5bjcr/1

By default, downloads ONLY the official KL grading subset
`KneeXrayData/ClsKLData/kneeKL224/{train,val,test}` (~180 MB) via a single
contiguous HTTP Range request from the official `KneeXrayData.zip` archive
(skipping ~6.2 GB of pretrained PyTorch checkpoints and knee-detection H5 files),
verifies every extracted PNG byte-for-byte against its official ZIP CRC32, and
organizes the untouched PNG images into `data/raw/KL_0/` .. `data/raw/KL_4/`.

Also supports extracting from a locally downloaded full `KneeXrayData.zip` via
`--local-zip`, verifying the full-archive SHA-256 against Mendeley Data's API.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import requests

DATASET_ID = "56rmx5bjcr"
DATASET_VERSION = 1
MENDELEY_API_META = (
    f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}?version={DATASET_VERSION}"
)
MENDELEY_API_FILES = (
    f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files"
    f"?folder_id=root&version={DATASET_VERSION}"
)
EXPECTED_ARCHIVE_NAME = "KneeXrayData.zip"
EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_GRADES = ("0", "1", "2", "3", "4")


class HTTPRangeReader(io.RawIOBase):
    """Minimal seekable read-only file over HTTP Range requests for reading ZIP central directory."""

    def __init__(self, url: str, total_size: int, session: requests.Session) -> None:
        self.url = url
        self.total_size = total_size
        self.pos = 0
        self.session = session

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.total_size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if self.pos < 0:
            raise ValueError("Negative seek position")
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n == -1 or self.pos + n > self.total_size:
            n = self.total_size - self.pos
        if n <= 0:
            return b""
        end = self.pos + n - 1
        resp = self.session.get(
            self.url,
            headers={"Range": f"bytes={self.pos}-{end}"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.content
        self.pos += len(data)
        return data


def fetch_mendeley_metadata(session: requests.Session) -> Tuple[dict, dict]:
    """Query Mendeley Data public API for verified dataset and file metadata."""
    meta_resp = session.get(MENDELEY_API_META, timeout=30)
    meta_resp.raise_for_status()
    meta = meta_resp.json()

    files_resp = session.get(MENDELEY_API_FILES, timeout=30)
    files_resp.raise_for_status()
    files_data = files_resp.json()

    files_list = files_data if isinstance(files_data, list) else files_data.get("value", [])
    target_file = None
    for item in files_list:
        if item.get("filename") == EXPECTED_ARCHIVE_NAME:
            target_file = item
            break
    if target_file is None:
        raise RuntimeError(
            f"Expected file {EXPECTED_ARCHIVE_NAME} not found in Mendeley API response."
        )
    return meta, target_file


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_member(filename: str, subset_prefix: str) -> Tuple[str, str, str] | None:
    """Return (source_split, kl_grade, file_name) for valid target entries, else None."""
    if not filename.startswith(subset_prefix):
        return None
    rel = filename[len(subset_prefix) :]
    parts = rel.split("/")
    if len(parts) != 3:
        return None
    source_split, kl_grade, fname = parts
    if source_split not in EXPECTED_SPLITS:
        return None
    if kl_grade not in EXPECTED_GRADES:
        return None
    if not fname or fname.endswith("/"):
        return None
    return source_split, kl_grade, fname


def extract_from_contiguous_range(
    session: requests.Session,
    download_url: str,
    subset: str,
    raw_dir: Path,
    tmp_dir: Path,
) -> Tuple[List[dict], dict]:
    """Download the exact contiguous byte slice for the selected subset and extract verified PNGs."""
    head = session.head(download_url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    final_url = head.url
    total_size = int(head.headers["Content-Length"])
    accept_ranges = head.headers.get("Accept-Ranges", "")
    if "bytes" not in accept_ranges.lower():
        raise RuntimeError(
            f"Server does not advertise byte-range support (Accept-Ranges={accept_ranges!r}). "
            "Use --local-zip with a manually downloaded KneeXrayData.zip instead."
        )

    print(f"[INFO] Reading ZIP central directory ({total_size:,} bytes archive)...")
    range_reader = HTTPRangeReader(final_url, total_size, session)
    zf = zipfile.ZipFile(range_reader)

    subset_prefix = f"KneeXrayData/ClsKLData/{subset}/"
    target_entries: List[Tuple[zipfile.ZipInfo, str, str, str]] = []
    excluded_summary: Dict[str, int] = {}

    for info in zf.infolist():
        if info.is_dir():
            continue
        parsed = parse_member(info.filename, subset_prefix)
        if parsed is not None:
            target_entries.append((info, *parsed))
        else:
            parts = info.filename.split("/")
            bucket = "/".join(parts[:4]) if len(parts) >= 4 else "/".join(parts[:-1])
            excluded_summary[bucket] = excluded_summary.get(bucket, 0) + 1

    if not target_entries:
        raise RuntimeError(f"No entries matched prefix {subset_prefix}")

    # Verify filename uniqueness before extraction
    seen_names: Dict[str, str] = {}
    for info, sp, kl, fname in target_entries:
        if fname in seen_names:
            raise RuntimeError(
                f"Filename collision detected: {fname} in {seen_names[fname]} and {info.filename}"
            )
        seen_names[fname] = info.filename

    # Compute contiguous byte span covering all target entries
    target_entries.sort(key=lambda item: item[0].header_offset)
    start_offset = target_entries[0][0].header_offset
    last_info = target_entries[-1][0]
    # Add 64 KiB safety margin for variable local header extra fields
    end_offset = min(
        total_size - 1,
        last_info.header_offset
        + 30
        + len(last_info.filename.encode("utf-8"))
        + len(last_info.extra)
        + last_info.compress_size
        + 65536,
    )
    span_bytes = end_offset - start_offset + 1
    print(
        f"[INFO] Selected {len(target_entries):,} PNG files from {subset_prefix} "
        f"(byte range {start_offset:,}..{end_offset:,}, {span_bytes / 1e6:.2f} MB)."
    )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    slice_path = tmp_dir / f"{subset}_slice.bin"

    # Stream contiguous range to disk
    print("[INFO] Streaming contiguous archive slice from Mendeley Data...")
    downloaded = 0
    with session.get(
        final_url,
        headers={"Range": f"bytes={start_offset}-{end_offset}"},
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        with slice_path.open("wb") as out_f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (25 * 1024 * 1024) < len(chunk):
                        pct = (downloaded / span_bytes) * 100
                        print(
                            f"       Downloaded {downloaded / 1e6:.1f} / {span_bytes / 1e6:.1f} MB ({pct:.1f}%)",
                            flush=True,
                        )
    if downloaded != span_bytes:
        raise RuntimeError(
            f"Incomplete slice download: expected {span_bytes:,} bytes, got {downloaded:,} bytes."
        )
    print(f"[INFO] Slice download complete ({downloaded / 1e6:.2f} MB). Extracting and verifying CRC32...")

    for g in EXPECTED_GRADES:
        (raw_dir / f"KL_{g}").mkdir(parents=True, exist_ok=True)

    index_rows: List[dict] = []
    with slice_path.open("rb") as sf:
        for info, source_split, kl_grade, fname in target_entries:
            rel_offset = info.header_offset - start_offset
            sf.seek(rel_offset)
            lfh = sf.read(30)
            if len(lfh) != 30 or lfh[:4] != b"PK\x03\x04":
                raise RuntimeError(
                    f"Invalid ZIP local file header signature for {info.filename} at offset {info.header_offset}"
                )
            fname_len, extra_len = struct.unpack("<HH", lfh[26:30])
            sf.seek(rel_offset + 30 + fname_len + extra_len)
            comp_data = sf.read(info.compress_size)
            if len(comp_data) != info.compress_size:
                raise RuntimeError(f"Short read for {info.filename}")

            if info.compress_type == zipfile.ZIP_STORED:
                raw_bytes = comp_data
            elif info.compress_type == zipfile.ZIP_DEFLATED:
                raw_bytes = zlib.decompress(comp_data, -15)
            else:
                raise RuntimeError(
                    f"Unsupported ZIP compression type {info.compress_type} for {info.filename}"
                )

            if len(raw_bytes) != info.file_size:
                raise RuntimeError(
                    f"Size mismatch for {info.filename}: expected {info.file_size}, got {len(raw_bytes)}"
                )
            actual_crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
            if actual_crc != info.CRC:
                raise RuntimeError(
                    f"CRC32 mismatch for {info.filename}: expected {info.CRC:08x}, got {actual_crc:08x}"
                )

            dest_rel = f"data/raw/KL_{kl_grade}/{fname}"
            dest_path = raw_dir / f"KL_{kl_grade}" / fname
            dest_path.write_bytes(raw_bytes)

            index_rows.append(
                {
                    "file_name": fname,
                    "relative_path": dest_rel,
                    "kl_grade": int(kl_grade),
                    "source_split": source_split,
                    "archive_member_path": info.filename,
                    "archive_crc32": f"{info.CRC:08x}",
                    "archive_file_size": info.file_size,
                }
            )

    # Remove temporary binary slice after full CRC32 verification
    slice_path.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    stats = {
        "download_method": "mendeley_official_http_range_slice",
        "byte_range_start": start_offset,
        "byte_range_end": end_offset,
        "downloaded_slice_bytes": span_bytes,
        "extracted_files": len(index_rows),
        "crc32_verified_files": len(index_rows),
        "excluded_archive_buckets": dict(sorted(excluded_summary.items())),
    }
    return index_rows, stats


def extract_from_local_zip(
    local_zip: Path,
    expected_sha256: str | None,
    subset: str,
    raw_dir: Path,
) -> Tuple[List[dict], dict]:
    """Verify and extract target subset from a locally downloaded KneeXrayData.zip."""
    if not local_zip.is_file():
        raise FileNotFoundError(f"Local ZIP not found: {local_zip}")

    if expected_sha256:
        print(f"[INFO] Computing SHA-256 of {local_zip}...")
        actual_sha256 = compute_sha256(local_zip)
        if actual_sha256.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"SHA-256 mismatch for {local_zip}: expected {expected_sha256}, got {actual_sha256}"
            )
        print("[INFO] Full-archive SHA-256 verified successfully.")

    for g in EXPECTED_GRADES:
        (raw_dir / f"KL_{g}").mkdir(parents=True, exist_ok=True)

    subset_prefix = f"KneeXrayData/ClsKLData/{subset}/"
    index_rows: List[dict] = []
    excluded_summary: Dict[str, int] = {}

    with zipfile.ZipFile(local_zip, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parsed = parse_member(info.filename, subset_prefix)
            if parsed is None:
                parts = info.filename.split("/")
                bucket = "/".join(parts[:4]) if len(parts) >= 4 else "/".join(parts[:-1])
                excluded_summary[bucket] = excluded_summary.get(bucket, 0) + 1
                continue
            source_split, kl_grade, fname = parsed
            raw_bytes = zf.read(info)
            actual_crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
            if actual_crc != info.CRC:
                raise RuntimeError(f"CRC32 mismatch for {info.filename}")
            dest_rel = f"data/raw/KL_{kl_grade}/{fname}"
            (raw_dir / f"KL_{kl_grade}" / fname).write_bytes(raw_bytes)
            index_rows.append(
                {
                    "file_name": fname,
                    "relative_path": dest_rel,
                    "kl_grade": int(kl_grade),
                    "source_split": source_split,
                    "archive_member_path": info.filename,
                    "archive_crc32": f"{info.CRC:08x}",
                    "archive_file_size": info.file_size,
                }
            )

    stats = {
        "download_method": "local_full_zip_extraction",
        "local_zip_path": str(local_zip),
        "extracted_files": len(index_rows),
        "crc32_verified_files": len(index_rows),
        "excluded_archive_buckets": dict(sorted(excluded_summary.items())),
    }
    return index_rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and organize Pingjun Chen's Mendeley Knee OA KL Grading Dataset."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the repository root.",
    )
    parser.add_argument(
        "--subset",
        choices=("kneeKL224", "kneeKL299"),
        default="kneeKL224",
        help="Official ClsKLData subset to extract (default: kneeKL224, author's primary benchmark).",
    )
    parser.add_argument(
        "--local-zip",
        type=Path,
        default=None,
        help="Optional path to a locally downloaded KneeXrayData.zip.",
    )
    args = parser.parse_args()

    project_root: Path = args.project_root.resolve()
    raw_dir = project_root / "data" / "raw"
    meta_dir = project_root / "data" / "metadata"
    tmp_dir = project_root / "data" / "tmp"
    meta_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "KneeOA-Dataset-Preparer/1.0"})

    print("[INFO] Fetching official metadata from Mendeley Data API...")
    meta, file_meta = fetch_mendeley_metadata(session)
    content_details = file_meta["content_details"]
    download_url = content_details["download_url"]
    expected_sha256 = content_details["sha256_hash"]

    if args.local_zip is not None:
        index_rows, dl_stats = extract_from_local_zip(
            args.local_zip, expected_sha256, args.subset, raw_dir
        )
    else:
        index_rows, dl_stats = extract_from_contiguous_range(
            session, download_url, args.subset, raw_dir, tmp_dir
        )

    # Sort deterministically and write source_index.csv
    index_rows.sort(key=lambda r: (r["kl_grade"], r["source_split"], r["file_name"]))
    index_csv = meta_dir / "source_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "relative_path",
                "kl_grade",
                "source_split",
                "archive_member_path",
                "archive_crc32",
                "archive_file_size",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    now_local = dt.datetime.now().astimezone().isoformat()
    contributors = [
        f"{c.get('first_name', '')} {c.get('last_name', '')}".strip()
        for c in meta.get("contributors", [])
    ]
    institutions = [inst.get("name", "") for inst in meta.get("institutions", [])]
    licence = meta.get("data_licence", {})

    provenance = {
        "dataset_name": meta.get("name"),
        "source_url": f"https://data.mendeley.com/datasets/{DATASET_ID}/{DATASET_VERSION}",
        "doi": meta.get("doi", {}).get("id"),
        "authors": contributors,
        "institutions": institutions,
        "version": meta.get("version"),
        "publish_date": meta.get("publish_date"),
        "description": meta.get("description", "").strip(),
        "origin_source": "Osteoarthritis Initiative (OAI) - https://oai.epi-ucsf.org/datarelease/",
        "author_code_repository": "https://github.com/PingjunChen/GradingKneeOA",
        "reference_paper": (
            "Chen, P., Gao, L., Shi, X., Allen, K., & Yang, L. (2019). "
            "Fully automatic knee osteoarthritis severity grading using deep neural networks "
            "with a novel ordinal loss. Computerized Medical Imaging and Graphics, 75, 84-92. "
            "https://doi.org/10.1016/j.compmedimag.2019.06.002"
        ),
        "license": {
            "short_name": licence.get("short_name"),
            "full_name": licence.get("full_name"),
            "url": licence.get("url"),
            "description": licence.get("description"),
        },
        "usage_limitations": [
            "Published on Mendeley Data under CC BY 4.0 with attribution to Pingjun Chen.",
            "Images originate from the Osteoarthritis Initiative (OAI); Mendeley's CC BY 4.0 notice explicitly states that further permission may be required for third-party content.",
            "Raw X-ray images are NOT committed to GitHub to respect repository size limits and third-party medical data redistribution practices.",
            "auto_test (1,526 YOLO-cropped test images) is excluded from manifest/split because it contains the same test subjects as the official test set.",
        ],
        "archive": {
            "filename": file_meta.get("filename"),
            "size_bytes": file_meta.get("size"),
            "sha256": expected_sha256,
            "download_url": download_url,
        },
        "selected_subset": {
            "path_in_archive": f"KneeXrayData/ClsKLData/{args.subset}/",
            "included_splits": list(EXPECTED_SPLITS),
            "excluded_splits_in_subset": ["auto_test"],
            "selection_rationale": (
                "kneeKL224 is the primary default dataset in Pingjun Chen's official ClsKL training "
                "code (train_kl.py) and standard benchmark literature. It contains 8,260 bilateral "
                "knee X-ray crops from 4,130 OAI baseline subjects partitioned by patient into "
                "train (5,778; 69.95%), val (826; 10.00%), and test (1,656; 20.05%). "
                "auto_test (1,526 images) is excluded because it consists of automatically detected "
                "crops of the exact same test subjects."
            ),
        },
        "download_timestamp_utc": now_utc,
        "download_timestamp_local": now_local,
        "download_verification": dl_stats,
    }

    prov_path = meta_dir / "source_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Extracted and verified {len(index_rows):,} images into {raw_dir}")
    print(f"[OK] Wrote {index_csv} and {prov_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
