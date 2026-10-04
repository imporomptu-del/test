"""Build the dependency-free native CUDA synthetic-tracking library."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).parent / "detection" / "cuda" / "synthetic_tracking.cu"
DEFAULT_OUTPUT = REPOSITORY / "build" / "cuda" / "libtiny_target_cuda.so"


def build_cuda_library(
    output: Path = DEFAULT_OUTPUT,
    *,
    nvcc: Path = Path("/usr/local/cuda/bin/nvcc"),
    architecture: str = "87",
) -> Path:
    if not architecture.isdigit() or len(architecture) not in {2, 3}:
        raise ValueError("CUDA architecture must look like 87 or 100")
    if not SOURCE.is_file():
        raise FileNotFoundError(f"CUDA source is missing: {SOURCE}")
    if not nvcc.is_file():
        raise FileNotFoundError(f"nvcc is missing: {nvcc}")
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    command = [
        str(nvcc),
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "--shared",
        "-Xcompiler=-fPIC",
        f"-gencode=arch=compute_{architecture},code=sm_{architecture}",
        str(SOURCE),
        "-o",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc")
    )
    parser.add_argument(
        "--architecture",
        default="87",
        help="CUDA SM architecture without the decimal point; AGX Orin is 87",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = build_cuda_library(
        args.output,
        nvcc=args.nvcc,
        architecture=args.architecture,
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Built {output}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
