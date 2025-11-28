#!/usr/bin/env python3
"""Download official pretrained TD-MPC2 checkpoints (excluding the ~1M model)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

try:
    import requests
except ImportError:  # pragma: no cover - fallback if requests unavailable
    requests = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tdmpc2.utils import DEFAULT_PRETRAINED_URLS  # noqa: E402


def _download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    if requests is None:
        raise ImportError(
            "The 'requests' package is required for downloading. Install it via pip or "
            "provide checkpoints manually."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def load_manifest(manifest_path: str | None) -> Dict[str, str]:
    if manifest_path is None:
        return dict(DEFAULT_PRETRAINED_URLS)
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {k.lower(): v for k, v in payload.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tdmpc2_pretrained",
        help="Directory to save downloaded checkpoints.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Optional JSON mapping of model size to download URL. If omitted, use "
            "built-in official URLs."
        ),
    )
    parser.add_argument(
        "--include_smallest",
        action="store_true",
        help="Include the ~1M checkpoint if present in the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if not args.include_smallest:
        manifest = {k: v for k, v in manifest.items() if not k.lower().startswith("1m")}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for size, url in manifest.items():
        filename = output_dir / f"tdmpc2_{size.lower()}.pt"
        print(f"Downloading {size} from {url} -> {filename}")
        try:
            _download_file(url, filename)
            downloaded.append((size, filename))
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  Failed to download {size} from {url}: {exc}")

    if downloaded:
        print("Downloaded checkpoints:")
        for size, path in downloaded:
            print(f"  {size}: {path}")
    else:
        print("No checkpoints downloaded. Verify manifest/URLs or network connectivity.")


if __name__ == "__main__":
    main()
