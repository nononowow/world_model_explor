#!/usr/bin/env python3
"""
Helper script to fetch the official World Models datasets (CarRacing + Doom).

Usage examples
--------------
Download both environments into the default layout:

    python scripts/download_worldmodels_data.py

Download only the Doom rollouts and place them under a custom directory:

    python scripts/download_worldmodels_data.py --env doom_take_cover --output-dir /tmp/worldmodels

By default the script skips files that are already present. Use `--force` to
re-download and overwrite existing artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen
import shutil
import zipfile

# ---------------------------------------------------------------------------
# Dataset specification

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "worldmodels"


class DatasetError(RuntimeError):
    """Custom exception for clearer user-facing error messages."""


# For each environment we define a tuple of (archive_name, download_url, optional_sha256)
DATASETS: Dict[str, Dict[str, Tuple[str, str, Optional[str]]]] = {
    "car_racing": {
        "observations.npy": ("carracing_obs.zip", "https://storage.googleapis.com/worldmodels/dataset/carracing_obs.zip", None),
        "actions.npy": ("carracing_act.zip", "https://storage.googleapis.com/worldmodels/dataset/carracing_act.zip", None),
        "rewards.npy": ("carracing_reward.zip", "https://storage.googleapis.com/worldmodels/dataset/carracing_reward.zip", None),
        "terminals.npy": ("carracing_terminal.zip", "https://storage.googleapis.com/worldmodels/dataset/carracing_terminal.zip", None),
    },
    "doom_take_cover": {
        "observations.npy": ("doom_obs.zip", "https://storage.googleapis.com/worldmodels/dataset/doom_obs.zip", None),
        "actions.npy": ("doom_act.zip", "https://storage.googleapis.com/worldmodels/dataset/doom_act.zip", None),
        "rewards.npy": ("doom_reward.zip", "https://storage.googleapis.com/worldmodels/dataset/doom_reward.zip", None),
        "terminals.npy": ("doom_terminal.zip", "https://storage.googleapis.com/worldmodels/dataset/doom_terminal.zip", None),
    },
}


# ---------------------------------------------------------------------------
# Helpers

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    env_names = sorted(DATASETS.keys())
    parser = argparse.ArgumentParser(
        description="Download the official World Models datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Available environments:
              - car_racing: OpenAI Gym CarRacing-v0 rollouts (RGB, continuous control)
              - doom_take_cover: VizDoom take_cover scenario (grayscale, discrete control)
            """
        ),
    )
    parser.add_argument(
        "--env",
        dest="envs",
        choices=env_names,
        action="append",
        help="Environment to download. Repeat to fetch multiple. Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Root directory where data is stored (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite files even if they already exist.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args(list(argv))


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def stream_download(url: str, destination: Path, *, quiet: bool = False) -> None:
    """Download a file with a simple progress indicator."""
    try:
        with urlopen(url) as response:
            total = response.length or 0
            chunk_size = 1024 * 1024
            downloaded = 0

            with destination.open("wb") as fh:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)

                    if not quiet and total:
                        percent = downloaded / total * 100
                        progress = f"{percent:6.2f}% ({human_bytes(downloaded)}/{human_bytes(total)})"
                        print(f"\r  ↳ downloading… {progress}", end="", flush=True)
            if not quiet and total:
                print()
    except URLError as error:
        raise DatasetError(f"Failed to download {url} ({error})") from error


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_single_file(zip_path: Path, destination: Path, *, quiet: bool = False) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise DatasetError(
                f"Expected a single file inside {zip_path.name}, found {len(members)} entries."
            )
        member_name = members[0]
        if not quiet:
            log(f"  ↳ extracting {member_name} → {destination.name}", quiet=quiet)
        with archive.open(member_name) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def download_and_extract(env: str, target_filename: str, file_spec: Tuple[str, str, str | None], target_dir: Path, *, force: bool, quiet: bool) -> None:
    archive_name, url, expected_hash = file_spec
    if not archive_name.endswith(".zip"):
        raise DatasetError(f"Expected .zip archive for {env}, got {archive_name}")

    destination = target_dir / target_filename

    if destination.exists() and not force:
        log(f"• {env}: {destination.name} already present, skipping.", quiet=quiet)
        return

    log(f"• {env}: fetching {url}", quiet=quiet)
    ensure_directory(target_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / archive_name
        stream_download(url, tmp_path, quiet=quiet)

        if expected_hash:
            digest = compute_sha256(tmp_path)
            if digest.lower() != expected_hash.lower():
                raise DatasetError(
                    f"Checksum mismatch for {archive_name}: expected {expected_hash}, got {digest}."
                )

        extract_single_file(tmp_path, destination, quiet=quiet)

    log(f"  ↳ stored at {destination}", quiet=quiet)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    environments = args.envs or sorted(DATASETS.keys())

    for env in environments:
        specs = DATASETS[env]
        env_dir = args.output_dir / env
        ensure_directory(env_dir)

        for target_filename, file_spec in specs.items():
            download_and_extract(
                env,
                target_filename,
                file_spec,
                env_dir,
                force=args.force,
                quiet=args.quiet,
            )

    log("All requested datasets are ready.", quiet=args.quiet)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except DatasetError as exc:
        print(f"[download_worldmodels_data] {exc}", file=sys.stderr)
        raise SystemExit(1)

