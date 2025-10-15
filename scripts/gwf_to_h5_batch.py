"""
Extract GW data from local frames using an FFL, VALIDATE channels via FrChannels by default,
and write an HDF5 with data and text reports.

Key features:
- **Channels Validation**: FrChannels-based validation runs first and only validated channels are extracted.
- **Batch-only extraction**: Uses a single batch read (TimeSeriesDict) with the LALFrame backend (format="gwf.lalframe").
  The per-channel extraction mode has been moved to an individual scripts (gwf_to_h5_per_channel.py).
- **Duration aliases**: `--duration` accepts seconds as a number, or strings with suffixes: "h" (hours) or "m" (minutes),
  e.g., `--duration 3600`, `--duration 1h`, `--duration 90m`, `--duration 1.5h`.
- **Channel list cleaning**: Automatically drops `*_500Hz` channels when using the "raw_full" FFL.
- **Output reports**: Generates text files listing valid and unavailable channels, as well as a summary report.

Outputs (all next to the HDF5 file unless unwritable, in which case current working dir is used):
- <name>_valid_channels.txt
- <name>_unavailable_channels.txt
- <name>_summary.txt
- <name>.h5
"""

import argparse
import os
import re
import subprocess
from pathlib import Path
from time import time
from typing import Iterable, Tuple, List, Set

from gwpy.time import to_gps
from gwdama.io.gwdatamanager import GwDataManager


# -----------------------------
# Helpers
# -----------------------------

def parse_ffl_arg(arg_ffl: str) -> Tuple[str, bool]:
    """Return (ffl_path, is_full). Supports 'raw' and 'raw_full' aliases."""
    is_full = arg_ffl == "raw_full" or arg_ffl == "/virgoData/ffl/raw_full.ffl"
    if os.path.isfile(arg_ffl):
        return arg_ffl, is_full
    if arg_ffl is None:
        raise ValueError("You must provide --ffl argument.")
    al = arg_ffl.lower()
    if al == "raw":
        return "/virgoData/ffl/raw.ffl", False
    if al == "raw_full":
        return "/virgoData/ffl/raw_full.ffl", True
    raise ValueError(f"Invalid --ffl argument: {arg_ffl}")


def parse_duration(s) -> float:
    """Parse duration that can be a number (seconds) or a string with 'h' or 'm' suffix.
    Examples: 60, "60", "1h", "1.5h", "90m", "3600s" (optional 's').
    Returns seconds as float.
    """
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower()
    # plain number -> seconds
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r'([0-9]*\.?[0-9]+)\s*([hms])', s)
    if not m:
        raise ValueError(f"Invalid --duration value: {s!r}. Use seconds (e.g. 3600) or add 'h'/'m' suffix (e.g. '1h', '90m').")
    value = float(m.group(1))
    unit = m.group(2)
    if unit == 'h':
        return value * 3600.0
    if unit == 'm':
        return value * 60.0
    # 's'
    return value


def load_channels(channels_file: str, drop_500Hz: bool=False, verbose: int=0) -> List[str]:
    """Read channels (one per line, '#' for comments), deduplicate, optionally drop *_500Hz."""
    chans = []
    with open(channels_file) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            chans.append(s)

    seen = set()
    uniq = []
    for c in chans:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
            if verbose >= 2:
                print(f"Added channel: {c}")

    if drop_500Hz:
        uniq = [c for c in uniq if not c.endswith("_500Hz")]
        if verbose >= 1:
            print(f"Dropped _500Hz channels, {len(uniq)} remain.")

    return uniq


def write_lists(valid: Iterable[str], missing: Iterable[str], base_dir: Path, base_name: str) -> Tuple[str, str]:
    """Write two lists (valid & unavailable) and return their paths as strings."""
    valid = list(valid)
    missing = list(missing)
    valid_path = Path(base_dir, f"{base_name}_valid_channels.txt")
    missing_path = Path(base_dir, f"{base_name}_unavailable_channels.txt")
    valid_path.write_text("\n".join(valid) + ("\n" if valid else ""))
    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""))
    return str(valid_path), str(missing_path)


def parse_ffl(ffl_path: str) -> List[Tuple[str, float, float]]:
    """Return list of (file, t0, t1) from an FFL."""
    entries = []
    with open(ffl_path) as f:
        for line in f:
            cols = line.split()
            if len(cols) < 3:
                continue
            try:
                file = cols[0]
                t0 = float(cols[1])
                dur = float(cols[2])
                entries.append((file, t0, t0 + dur))
            except Exception:
                continue
    return entries


def overlapping_files(ffl_entries: List[Tuple[str, float, float]], t0: float, t1: float) -> List[str]:
    """Filter FFL entries that overlap [t0, t1)."""
    return [f for (f, a, b) in ffl_entries if not (b <= t0 or a >= t1)]


def frchannels_list(frchannels_path: str, file_path: str) -> Set[str]:
    """Run FrChannels on a single frame file and return a set of channel names."""
    attempts = (
        [frchannels_path, "-l", file_path],
        [frchannels_path, file_path],
    )
    last_err = None
    for cmd in attempts:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            chans: Set[str] = set()
            for line in out.splitlines():
                s = line.strip()
                if not s:
                    continue
                if s.startswith("Usage:") or s.startswith("This utility"):
                    continue
                name = s.split()[0]
                if name:
                    chans.add(name)
            if chans:
                return chans
        except subprocess.CalledProcessError as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("FrChannels invocation failed")


def validate_channels(ffl_path: str, start_gps: float, end_gps: float, requested: List[str],
                      frchannels_path: str = "FrChannels", scan_limit: int = 3, verbose: int = 0
                      ) -> Tuple[List[str], List[str]]:
    """Centralized validation: scan a limited number of overlapping frames with FrChannels and
    return (valid, missing) preserving requested order.
    """
    entries = parse_ffl(ffl_path)
    files = overlapping_files(entries, start_gps, end_gps)
    if scan_limit and scan_limit > 0:
        files = files[:scan_limit]

    inventory: Set[str] = set()
    for fp in files:
        if Path(fp).is_file():
            try:
                inv = frchannels_list(frchannels_path, fp)
                inventory |= inv
                if verbose >= 2:
                    print(f"\nScanned {fp}: +{len(inv)} channels (union={len(inventory)})")
            except subprocess.CalledProcessError as e:
                if verbose >= 1:
                    print(f"\n[warn] FrChannels failed on {fp}: {e}")
            except Exception as e:
                if verbose >= 1:
                    print(f"\n[warn] Skipping {fp}: {e}")

    # preserve order of requested
    req_ordered = list(dict.fromkeys(requested))
    valid = [ch for ch in req_ordered if ch in inventory]
    missing = [ch for ch in req_ordered if ch not in inventory]
    return valid, missing


def extract_batch(channels: List[str], ffl_path: str, start_gps: float, end_gps: float,
                  out_h5: Path, nproc: int, verbose: int) -> Tuple[List[str], List[str]]:
    """Batch extraction using LALFrame. Returns (available, unavailable_from_extraction)."""
    available: List[str] = []
    unavailable: List[str] = []

    if not channels:
        # No channels to extract after validation.
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        GwDataManager().write_gwdama(filename=str(out_h5))  # empty container
        return available, unavailable

    read_kwargs = {
        "start": start_gps,
        "end": end_gps,
        "nproc": nproc,
        "crop": True,
        "ffl_spec": None,
        "ffl_path": ffl_path,
        "format": "gwf.lalframe",
    }

    dama = GwDataManager()  # in-memory
    try:
        tsd = GwDataManager.read_from_virgo(channels=channels, **read_kwargs)
        if hasattr(tsd, "items"):  # TimeSeriesDict path
            dama.from_TimeSeries(tsd, dts_key="channels", compression="gzip")
            got = list(tsd.keys())
            available.extend(got)
            missing_set = set(channels) - set(got)
            unavailable.extend(sorted(missing_set))
            if verbose >= 1:
                print("\n[batch] Batch read (lalframe) succeeded.")
        else:
            # Single-TS path (shouldn't happen if len(channels) > 1)
            dama.from_TimeSeries(tsd, dts_key=channels[0], compression="gzip")
            available.append(channels[0])
            if verbose >= 1:
                print("\n[batch] Batch read (lalframe) returned single series.")
    except Exception as e:
        if verbose >= 1:
            print(f"\n[batch] Batch read failed: {e}")
        # If batch fails entirely, mark all requested as unavailable_from_extraction
        unavailable.extend(channels)

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    dama.write_gwdama(filename=str(out_h5))
    dama.close()
    return available, unavailable


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract GW data to HDF5 with default FrChannels validation and LALFrame batch reads."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ffl", help="Path to an FFL file to locate GWFs. Use 'raw' or 'raw_full' aliases.", default="raw")

    p.add_argument("--start", required=True,
                   help="Start time (GPS number or 'YYYY-MM-DD [HH:MM:SS]').")
    p.add_argument("--duration", required=True,
                   help="Duration in seconds, or string with suffix: 'h' for hours or 'm' for minutes (e.g., '1h', '90m').")
    p.add_argument("--channels-file", help="Text file with one channel per line (comments with #).", required=True)
    p.add_argument("--nproc", type=int, default=1, help="Processes for GWpy read (default: 1).")
    p.add_argument("--out", required=True, help="Output HDF5 file path.")

    # Validation tuning (still configurable, but always performed)
    p.add_argument("--scan-limit", type=int, default=3,
                   help="Max number of overlapping frames to scan with FrChannels (union). 0=all.")
    p.add_argument("--frchannels-path", default="FrChannels",
                   help="Path to FrChannels binary (default: 'FrChannels').")

    p.add_argument("--verbose", default=0, type=int, help="Verbosity level (0=quiet).")

    return p.parse_args()


def main():
    args = parse_args()
    t_start = time()

    # Resolve FFL
    ffl_path, is_full = parse_ffl_arg(args.ffl)

    # Channels
    channels_all = load_channels(args.channels_file, drop_500Hz=is_full, verbose=args.verbose)

    if args.verbose >= 1:
        print("\nArguments:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

    # Times
    start_gps = float(to_gps(args.start))
    duration_sec = parse_duration(args.duration)
    end_gps = start_gps + duration_sec

    # Output paths
    out_h5 = Path(args.out).expanduser().resolve()
    base_dir = out_h5.parent
    base_name = out_h5.stem
    base_dir.mkdir(parents=True, exist_ok=True)

    # ----- Validation (ALWAYS ON) -----
    valid, missing_from_validation = validate_channels(
        ffl_path=ffl_path,
        start_gps=start_gps,
        end_gps=end_gps,
        requested=channels_all,
        frchannels_path=args.frchannels_path,
        scan_limit=args.scan_limit,
        verbose=args.verbose,
    )
    valid_list_path, missing_list_path = write_lists(sorted(valid), sorted(missing_from_validation), base_dir, base_name)

    # ----- Extraction (Batch-only) -----
    available, unavailable_from_extraction = extract_batch(
        channels=valid,
        ffl_path=ffl_path,
        start_gps=start_gps,
        end_gps=end_gps,
        out_h5=out_h5,
        nproc=args.nproc,
        verbose=args.verbose,
    )

    # Merge missing sets (validation-missing + extraction-missing)
    combined_missing = sorted(set(missing_from_validation) | set(unavailable_from_extraction))

    # Re-write lists to reflect actual availability after extraction
    valid_list_path, missing_list_path = write_lists(sorted(available), combined_missing, base_dir, base_name)

    # ----- Summary -----
    elapsed = time() - t_start
    size_mb = out_h5.stat().st_size / (1024**2) if out_h5.exists() else 0.0

    if args.verbose >= 1:
        print("\nSummary:")
        print(f"- Requested channels: {len(channels_all)}")
        print(f"- Validated channels: {len(valid)}")
        print(f"- Available (written): {len(available)} (see {valid_list_path})")
        print(f"- Unavailable: {len(combined_missing)} (see {missing_list_path})")
        print(f"- Size on disk: {size_mb:.2f} MB")
        print(f"- Output HDF5 file: {out_h5}")
        print(f"- Elapsed time: {elapsed:.2f} s")

    summary_path = Path(base_dir, f"{base_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Summary of GW data extraction (validation + batch extraction)\n")
        f.write("=============================================================\n\n")
        f.write(f"Start time (GPS): {start_gps}\n")
        f.write(f"End time (GPS): {end_gps}\n")
        f.write(f"Duration (s): {duration_sec}\n")
        f.write(f"Processes used: {args.nproc}\n")
        f.write(f"FFL used: {ffl_path}\n")
        f.write(f"Channels file: {args.channels_file}\n")
        f.write(f"Validated channels: {len(valid)}\n")
        f.write(f"Available channels written: {len(available)}  ({valid_list_path})\n")
        f.write(f"Unavailable channels: {len(combined_missing)}  ({missing_list_path})\n")
        f.write(f"Output HDF5 file: {out_h5}  ({size_mb:.2f} MB)\n")
        f.write(f"Elapsed time (s): {elapsed:.2f}\n")


if __name__ == "__main__":
    main()
