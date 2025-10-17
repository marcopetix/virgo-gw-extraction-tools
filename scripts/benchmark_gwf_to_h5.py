
#!/usr/bin/env python3
"""
Benchmark runner for gwf_to_h5_batch.py

Runs multiple extractions for different channel lists and nproc values,
repeating each combination multiple times, then reports mean/std elapsed time.

Default scenario (matches the user's request):
- Date/time: 2025-10-07 00:00:00
- Duration: 1 minute
- Channel files: all_channels.txt, aux_channels.txt, Sa_channels.txt, Sc_channels.txt
- nproc: 1..8
- Repeats: 3
- Overwrites the same output file per (channel-tag), as requested.
- Produces a CSV with all timings and one with aggregated stats, and prints a table.

Usage example:
  python benchmark_gwf_to_h5.py \
    --script-path scripts/gwf_to_h5_batch.py \
    --ffl raw \
    --start "2025-10-07 00:00:00" \
    --duration 1m \
    --channels-dir channels \
    --out-root /data/procdata/rcsDatasets/OriginalSR/datasets_h5_petix \
    --repeats 3 \
    --nproc-min 1 --nproc-max 8 \
    --scan-limit 3 \
    --verbose 1

Note: elapsed time is measured externally (wall-clock) around the subprocess.
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------

def channel_tag_from_filename(filename: str) -> str:
    """
    Create a short tag for the output path from the channel filename.
    E.g. 'Sa_channels.txt' -> 'Sa', 'aux_channels.txt' -> 'aux', 'all_channels.txt' -> 'all', 'Sc_channels.txt' -> 'Sc'.
    """
    base = Path(filename).stem  # e.g., 'Sa_channels'
    # Strip common suffixes
    for suffix in ("_channels", "-channels", "_chans", "-chans"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    # Special-case 'all' from 'all_channels'
    return base


def mk_out_path(out_root: Path, tag: str, date_tag: str = "2025_10_07") -> Path:
    """
    Match the user's directory structure:
      /data/.../datasets_h5_petix/{TAG}_2025_10_07/dataset.h5
    """
    return out_root / f"{tag}_{date_tag}" / "dataset.h5"


@dataclass
class RunResult:
    channels_file: str
    tag: str
    nproc: int
    repeat_idx: int
    returncode: int
    elapsed_s: Optional[float]
    out_file: str


def run_once(
    script_path: Path,
    ffl: str,
    start: str,
    duration: str,
    channels_file: Path,
    nproc: int,
    out_file: Path,
    scan_limit: int,
    frchannels_path: str,
    verbose: int,
) -> RunResult:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script_path),
        "--ffl", ffl,
        "--start", start,
        "--duration", duration,
        "--channels-file", str(channels_file),
        "--out", str(out_file),
        "--nproc", str(nproc),
        "--scan-limit", str(scan_limit),
        "--frchannels-path", frchannels_path,
        "--verbose", str(verbose),
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=(verbose == 0), text=True)
        rc = proc.returncode
        if verbose >= 2 and proc.stdout:
            print(proc.stdout)
        if verbose >= 1 and proc.stderr:
            print(proc.stderr, file=sys.stderr)
    except Exception as e:
        if verbose >= 1:
            print(f"[error] Subprocess failed to start/run: {e}", file=sys.stderr)
        return RunResult(
            channels_file=str(channels_file),
            tag=channel_tag_from_filename(channels_file.name),
            nproc=nproc,
            repeat_idx=-1,
            returncode=-1,
            elapsed_s=None,
            out_file=str(out_file),
        )
    elapsed = time.perf_counter() - t0

    return RunResult(
        channels_file=str(channels_file),
        tag=channel_tag_from_filename(channels_file.name),
        nproc=nproc,
        repeat_idx=-1,  # will be set by caller
        returncode=rc,
        elapsed_s=elapsed if rc == 0 else None,
        out_file=str(out_file),
    )


def summarize(rows: List[RunResult]) -> List[dict]:
    """
    Aggregate by (tag, nproc). Compute mean and population std dev over successful runs.
    """
    agg = {}
    for r in rows:
        key = (r.tag, r.nproc)
        agg.setdefault(key, []).append(r)

    summary = []
    for (tag, nproc), group in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        times = [g.elapsed_s for g in group if g.elapsed_s is not None]
        successes = sum(1 for g in group if g.returncode == 0 and g.elapsed_s is not None)
        failures = len(group) - successes
        mean_s = mean(times) if times else math.nan
        std_s = pstdev(times) if len(times) > 1 else 0.0 if len(times) == 1 else math.nan
        summary.append({
            "tag": tag,
            "nproc": nproc,
            "runs": len(group),
            "successes": successes,
            "failures": failures,
            "mean_s": mean_s,
            "std_s": std_s,
        })
    return summary


def print_table(summary: List[dict]) -> None:
    # Determine nproc range and tags
    tags = sorted(set(item["tag"] for item in summary))
    nprocs = sorted(set(item["nproc"] for item in summary))

    # Build a grid: rows=tags, columns=nproc -> "mean ± std (succ/total)"
    col_width = 18
    header = "tag \\ nproc".ljust(12) + "".join(f"{n:>{col_width}}" for n in nprocs)
    print("\n" + header)
    print("-" * len(header))
    for tag in tags:
        row = tag.ljust(12)
        for n in nprocs:
            cell_items = [s for s in summary if s["tag"] == tag and s["nproc"] == n]
            if not cell_items:
                cell = "—"
            else:
                s = cell_items[0]
                if math.isnan(s["mean_s"]):
                    cell = "NaN"
                else:
                    cell = f"{s['mean_s']:.2f} ± {s['std_s']:.2f} ({s['successes']}/{s['runs']})"
            row += f"{cell:>{col_width}}"
        print(row)
    print()


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Benchmark gwf_to_h5_batch.py across channels and nproc.")
    ap.add_argument("--script-path", default="scripts/gwf_to_h5_batch.py",
                    help="Path to gwf_to_h5_batch.py (default: scripts/gwf_to_h5_batch.py)")
    ap.add_argument("--ffl", default="raw", help="FFL spec or path (default: raw)")
    ap.add_argument("--start", default="2025-10-07 00:00:00", help="Start datetime or GPS (default: 2025-10-07 00:00:00)")
    ap.add_argument("--duration", default="1m", help="Duration (e.g., 60, 1m, 1h) (default: 1m)")

    ap.add_argument("--channels-dir", default="channels", help="Directory containing channel lists (default: channels)")
    ap.add_argument("--channels", nargs="*", default=["all_channels.txt", "aux_channels.txt", "Sa_channels.txt", "Sc_channels.txt"],
                    help="Channel list filenames to test (default: all_channels.txt aux_channels.txt Sa_channels.txt Sc_channels.txt)")

    ap.add_argument("--out-root", default="/data/procdata/rcsDatasets/OriginalSR/datasets_h5_petix",
                    help="Root directory for output HDF5s (default as requested)")

    ap.add_argument("--repeats", type=int, default=3, help="Number of runs per (channels, nproc) (default: 3)")
    ap.add_argument("--nproc-min", type=int, default=1, help="Minimum nproc (default: 1)")
    ap.add_argument("--nproc-max", type=int, default=8, help="Maximum nproc (default: 8)")

    ap.add_argument("--scan-limit", type=int, default=3, help="FrChannels scan limit (default: 3)")
    ap.add_argument("--frchannels-path", default="FrChannels", help="Path to FrChannels binary (default: FrChannels)")

    ap.add_argument("--verbose", type=int, default=1, help="Verbosity for child script (0,1,2) (default: 1)")

    ap.add_argument("--results-dir", default="benchmark_results",
                    help="Where to write CSV/MD summaries (default: benchmark_results)")

    args = ap.parse_args()

    script_path = Path(args.script_path).expanduser().resolve()
    if not script_path.exists():
        print(f"[fatal] Script not found at: {script_path}", file=sys.stderr)
        sys.exit(2)

    channels_dir = Path(args.channels_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()

    channels_files = [channels_dir / c for c in args.channels]
    for cf in channels_files:
        if not cf.exists():
            print(f"[fatal] Channel file not found: {cf}", file=sys.stderr)
            sys.exit(2)

    nprocs = list(range(args.nproc_min, args.nproc_max + 1))

    print("Benchmark configuration")
    print("=======================")
    print(f"script_path : {script_path}")
    print(f"ffl         : {args.ffl}")
    print(f"start       : {args.start}")
    print(f"duration    : {args.duration}")
    print(f"channels    : {[str(c) for c in channels_files]}")
    print(f"nprocs      : {nprocs}")
    print(f"repeats     : {args.repeats}")
    print(f"out_root    : {out_root}")
    print(f"results_dir : {results_dir}")
    print()

    all_rows: List[RunResult] = []

    for ch_file in channels_files:
        tag = channel_tag_from_filename(ch_file.name)
        out_file = mk_out_path(out_root, tag=tag, date_tag="2025_10_07")

        for nproc in nprocs:
            for r in range(args.repeats):
                print(f"[run] tag={tag:>3}  nproc={nproc}  repeat={r+1}/{args.repeats}")
                res = run_once(
                    script_path=script_path,
                    ffl=args.ffl,
                    start=args.start,
                    duration=args.duration,
                    channels_file=ch_file,
                    nproc=nproc,
                    out_file=out_file,
                    scan_limit=args.scan_limit,
                    frchannels_path=args.frchannels_path,
                    verbose=args.verbose,
                )
                res.repeat_idx = r
                all_rows.append(res)

    # Save raw rows
    rows_dicts = [{
        "channels_file": rr.channels_file,
        "tag": rr.tag,
        "nproc": rr.nproc,
        "repeat_idx": rr.repeat_idx,
        "returncode": rr.returncode,
        "elapsed_s": f"{rr.elapsed_s:.6f}" if rr.elapsed_s is not None else "",
        "out_file": rr.out_file,
    } for rr in all_rows]

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = results_dir / "timings_raw.csv"
    write_csv(raw_csv, rows_dicts, fieldnames=list(rows_dicts[0].keys()))
    print(f"[ok] Wrote raw timings to: {raw_csv}")

    # Aggregates
    summary = summarize(all_rows)
    summary_csv = results_dir / "timings_summary.csv"
    write_csv(summary_csv, summary, fieldnames=["tag", "nproc", "runs", "successes", "failures", "mean_s", "std_s"])
    print(f"[ok] Wrote summary to: {summary_csv}")

    # Markdown table for convenience
    md_path = results_dir / "timings_summary.md"
    with open(md_path, "w") as f:
        f.write("# Benchmark summary (mean ± std, successes/total)\n\n")
        # Render a simple matrix
        tags = sorted(set(item["tag"] for item in summary))
        nprocs = sorted(set(item["nproc"] for item in summary))
        # Header
        f.write("| tag \\ nproc | " + " | ".join(str(n) for n in nprocs) + " |\n")
        f.write("|---|" + "|".join("---" for _ in nprocs) + "|\n")
        for tag in tags:
            row_cells = []
            for n in nprocs:
                found = [s for s in summary if s["tag"] == tag and s["nproc"] == n]
                if not found:
                    cell = "—"
                else:
                    s = found[0]
                    if math.isnan(s["mean_s"]):
                        cell = "NaN"
                    else:
                        cell = f"{s['mean_s']:.2f} ± {s['std_s']:.2f} ({s['successes']}/{s['runs']})"
                row_cells.append(cell)
            f.write(f"| {tag} | " + " | ".join(row_cells) + " |\n")
    print(f"[ok] Wrote markdown table to: {md_path}\n")

    # Print a pretty table to stdout as requested
    print_table(summary)


if __name__ == "__main__":
    main()
