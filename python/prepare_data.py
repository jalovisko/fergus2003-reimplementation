#!/usr/bin/env python3
"""Populate data/ with JPEGs for main_fergus2003.py from Caltech-hosted archives (stdlib only).

Not byte-identical to the paper's exact 2003 file lists: Caltech-101 is the same
broad lineage, but counts differ from older standalone web-harvest zips.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

CALTECH101_ZIP = (
    "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
)
CARS2001_ZIP = (
    "https://data.caltech.edu/records/dvx6b-vsc46/files/Cars_2001.zip?download=1"
)

UA = "fergus2003-reimplementation/prepare_data (Python urllib)"


def _default_data_dir() -> Path:
    """Same default layout as main_fergus2003.py when run from python/."""
    return Path(__file__).resolve().parent / "data"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing archive {dest}")
        return
    print(f"Downloading\n  {url}\n  -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def _find_101_object_categories_dir(root: Path) -> Path | None:
    """Return .../101_ObjectCategories if it contains airplanes/, else None."""
    for p in root.rglob("101_ObjectCategories"):
        if "__MACOSX" in p.parts:
            continue
        if p.is_dir() and (p / "airplanes").is_dir():
            return p
    return None


def _resolve_101_object_categories(extract_dir: Path) -> Path:
    """Caltech's zip ships 101_ObjectCategories.tar.gz; older mirrors had a plain folder."""
    found = _find_101_object_categories_dir(extract_dir)
    if found is not None:
        return found
    tarballs = sorted(
        p
        for p in extract_dir.rglob("101_ObjectCategories.tar.gz")
        if p.is_file() and "__MACOSX" not in p.parts
    )
    if not tarballs:
        raise FileNotFoundError(
            f"No 101_ObjectCategories/ or 101_ObjectCategories.tar.gz under {extract_dir}"
        )
    tar_path = tarballs[0]
    untar = extract_dir / "_101_ObjectCategories_unpacked"
    if untar.exists():
        shutil.rmtree(untar)
    untar.mkdir(parents=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(untar)
    found = _find_101_object_categories_dir(untar)
    if found is None:
        raise FileNotFoundError(
            f"Extracted {tar_path.name} but no 101_ObjectCategories/airplanes under {untar}"
        )
    return found


def _copy_images(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for f in sorted(src_dir.glob(pattern)):
            if f.is_file():
                shutil.copy2(f, dst_dir / f.name)
                n += 1
    return n


def _prepare_caltech101(
    data_dir: Path, work_dir: Path, face_category: str, clean_targets: bool
) -> None:
    arch = work_dir / "caltech-101.zip"
    extract_dir = work_dir / "caltech-101_extracted"
    _download(CALTECH101_ZIP, arch)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(arch, "r") as zf:
        zf.extractall(extract_dir)

    root101 = _resolve_101_object_categories(extract_dir)
    mapping = {
        "airplanes": "airplanes",
        "Motorbikes": "motorbikes",
        face_category: "faces",
        "BACKGROUND_Google": "background",
    }
    for src_name, dst_name in mapping.items():
        src = root101 / src_name
        if not src.is_dir():
            raise FileNotFoundError(f"Missing Caltech-101 folder: {src}")
        dst = data_dir / dst_name
        if clean_targets and dst.exists():
            shutil.rmtree(dst)
        count = _copy_images(src, dst)
        print(f"  {dst_name}: copied {count} images from {src_name}/")


def _prepare_cars2001(data_dir: Path, work_dir: Path, clean_targets: bool) -> None:
    arch = work_dir / "Cars_2001.zip"
    extract_dir = work_dir / "cars_2001_extracted"
    _download(CARS2001_ZIP, arch)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(arch, "r") as zf:
        zf.extractall(extract_dir)

    # Zip contains Cars_2001/cars_brad.tar with JPEGs at archive root (not loose files).
    car_tars = sorted(
        p
        for p in extract_dir.rglob("cars_brad.tar")
        if p.is_file() and "__MACOSX" not in p.parts
    )
    if not car_tars:
        car_tars = sorted(
            p
            for p in extract_dir.rglob("*.tar")
            if p.is_file() and "__MACOSX" not in p.parts
        )
    if not car_tars:
        raise FileNotFoundError(
            f"No cars_brad.tar or other .tar with images under {extract_dir}"
        )
    tar_path = car_tars[0]
    untar = extract_dir / "_cars_unpacked"
    if untar.exists():
        shutil.rmtree(untar)
    untar.mkdir(parents=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(untar)

    dst = data_dir / "cars_rear"
    if clean_targets and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(untar.rglob("*")):
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy2(f, dst / f.name)
            n += 1
    print(f"  cars_rear: copied {n} images from {tar_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Output directory (default: python/data, same as main_fergus2003.py)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".download_cache",
        help="Where to store zip archives and temp extract (default: python/.download_cache)",
    )
    p.add_argument(
        "--face-category",
        choices=("Faces_easy", "Faces"),
        default="Faces_easy",
        help="Caltech-101 face folder to map to data/faces/ (default: Faces_easy)",
    )
    p.add_argument(
        "--skip-caltech101",
        action="store_true",
        help="Advanced: skip Caltech-101 copy (only unpack Cars 2001 → cars_rear/)",
    )
    p.add_argument(
        "--skip-cars",
        action="store_true",
        help="Advanced: skip Cars 2001 (only Caltech-101 → airplanes, motorbikes, faces, background)",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove each target class folder before copying into it",
    )
    args = p.parse_args()
    data_dir = args.data_dir.resolve()
    work_dir = args.cache_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_caltech101:
        print("Caltech-101 (airplanes, motorbikes, faces, background)…")
        _prepare_caltech101(data_dir, work_dir, args.face_category, args.clean)
    if not args.skip_cars:
        print("Caltech Cars 2001 (rear)…")
        _prepare_cars2001(data_dir, work_dir, args.clean)
    print("Done.")


if __name__ == "__main__":
    main()
