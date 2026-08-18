#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Push a Kaggle kernel, run it on Kaggle compute, and fetch outputs and log.

Remote-execution loop for environments without local GPUs: edit code locally,
push it as a kernel, let Kaggle run it, then download the run log and output
files for inspection. Never submits to a competition — use submit_kernel.py
for that.

Usage:
    python run_kernel.py <kernel-folder> [--output DIR]
           [--poll-interval SEC] [--timeout SEC] [--no-download] [--log-tail N]
"""

import argparse
import os
import sys
from pathlib import Path

# constants and submit_kernel live in the sibling base skill.
sys.path.append(str(Path(__file__).resolve().parents[2] / "nvidia-kaggle-skill" / "scripts"))

from constants import (  # noqa: E402
    DEFAULT_KERNEL_TIMEOUT_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    OUTPUT_SEPARATOR_WIDTH,
)
from submit_kernel import (
    format_duration,
    get_api,
    has_kaggle_credentials,
    poll_kernel,
    push_kernel,
    read_kernel_metadata,
)

OUTPUT_SEPARATOR = "=" * OUTPUT_SEPARATOR_WIDTH
DEFAULT_LOG_TAIL_LINES = 50


def download_outputs(api, slug: str, out_dir: str) -> list[str]:
    """Download kernel output files and run log into out_dir. Returns file list."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nDownloading outputs to {out_dir} ...")
    try:
        api.kernels_output(slug, path=out_dir, force=True, quiet=False)
    except Exception as e:
        print(f"Output download failed: {e}", file=sys.stderr)
        return []
    files = []
    for root, _dirs, names in os.walk(out_dir):
        for name in names:
            files.append(os.path.relpath(os.path.join(root, name), out_dir))
    return sorted(files)


def print_log_tail(out_dir: str, slug: str, lines: int) -> None:
    """Print the last N lines of the downloaded kernel run log, if present."""
    kernel_name = slug.split("/")[-1]
    log_path = os.path.join(out_dir, f"{kernel_name}.log")
    if not os.path.exists(log_path):
        print(f"Log:         not found at {log_path}")
        return
    with open(log_path, encoding="utf-8", errors="replace") as f:
        tail = f.readlines()[-lines:]
    print(f"\nLog tail ({log_path}, last {len(tail)} lines):")
    print("-" * OUTPUT_SEPARATOR_WIDTH)
    for line in tail:
        print(line.rstrip())
    print("-" * OUTPUT_SEPARATOR_WIDTH)


def main():
    parser = argparse.ArgumentParser(description="Run a kernel on Kaggle compute and fetch results.")
    parser.add_argument("path", help="Path to kernel folder (must contain kernel-metadata.json)")
    parser.add_argument("--output", help="Directory for downloaded outputs and log (default: <kernel-folder>/output)")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help=f"Seconds between status checks (default: {DEFAULT_POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_KERNEL_TIMEOUT_SECONDS,
        help=f"Max seconds to wait (default: {DEFAULT_KERNEL_TIMEOUT_SECONDS} = 24h)",
    )
    parser.add_argument("--no-download", action="store_true", help="Skip downloading outputs and log")
    parser.add_argument(
        "--log-tail",
        type=int,
        default=DEFAULT_LOG_TAIL_LINES,
        help=f"Lines of run log to print (default: {DEFAULT_LOG_TAIL_LINES}, 0 to disable)",
    )
    args = parser.parse_args()

    if not has_kaggle_credentials():
        print("Error: No Kaggle credentials found.\n"
              "Set KAGGLE_API_TOKEN environment variable.",
              file=sys.stderr)
        sys.exit(1)

    kernel_path = os.path.abspath(args.path)
    if not os.path.isdir(kernel_path):
        print(f"Error: '{kernel_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    meta = read_kernel_metadata(kernel_path)
    slug = meta["id"]
    gpu = meta.get("enable_gpu", False)
    internet = meta.get("enable_internet", False)
    datasets = meta.get("dataset_sources", [])
    competitions = meta.get("competition_sources", [])
    kernels = meta.get("kernel_sources", [])

    print(f"Kernel:      {slug}")
    print(f"Code file:   {meta['code_file']}")
    print(f"GPU:         {'yes' if gpu else 'no'}")
    print(f"Internet:    {'yes' if internet else 'no'}")
    if datasets:
        print(f"Datasets:    {', '.join(datasets)}")
    if competitions:
        print(f"Competition: {', '.join(competitions)}")
    if kernels:
        print(f"Kernels:     {', '.join(kernels)}")

    api = get_api()
    version = push_kernel(api, kernel_path)
    status, elapsed = poll_kernel(api, slug, args.poll_interval, args.timeout)

    print(f"\n{OUTPUT_SEPARATOR}")
    print(f"Kernel:      {slug}")
    print(f"Version:     {version}")
    print(f"Status:      {status}")
    print(f"Runtime:     {format_duration(elapsed)} (±{args.poll_interval}s)")

    if args.no_download:
        print("Outputs:     skipped (--no-download)")
    elif status == "timeout":
        print("Outputs:     skipped (kernel still running; rerun with --timeout or fetch later\n"
              f"             with: kaggle kernels output {slug} -p <dir>)")
    else:
        out_dir = os.path.abspath(args.output or os.path.join(kernel_path, "output"))
        files = download_outputs(api, slug, out_dir)
        if files:
            print(f"Outputs ({len(files)} files):")
            for f in files:
                print(f"  {f}")
        else:
            print("Outputs:     none downloaded")
        if args.log_tail > 0:
            print_log_tail(out_dir, slug, args.log_tail)

    print(OUTPUT_SEPARATOR)

    if status != "complete":
        sys.exit(1)


if __name__ == "__main__":
    main()
