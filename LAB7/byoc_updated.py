#!/usr/bin/env python3
"""
Fetch MiniSEED waveform windows for aircraft-station crossing times.

This script reads a crossing file containing station names and closest-approach
times, downloads waveform windows from IRIS, and saves them as MiniSEED files.

The code can run:
1. Locally on a single machine
2. In parallel on an HPC system using SLURM array tasks

Inputs
------
- Crossing file in CSV-like format with no header
- Station/network/channel configuration defined in NETWORK_CONFIGS

Outputs
-------
- MiniSEED waveform files saved under:
  OUTPUT_ROOT / network / station / filename.mseed

Example
-------
Local run:
    python fetch_waveforms.py --crossing-file crossings_filtered.txt

SLURM array run:
    sbatch --array=0-9 run_fetch_waveforms.sh

Requirements
------------
- pandas
- obspy
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


DEFAULT_CROSSING_FILE = "/Users/sushmitamaurya/parkshwynodal/local_data_fetch/crossings_filtered.txt"
DEFAULT_OUTPUT_ROOT = Path.home() / "Desktop" / "ALL_NETWORKS_WAVEFORMS"
DEFAULT_TIME_WINDOW = 120.0
DEFAULT_SLEEP_SECONDS = 0.1

NETWORK_CONFIGS = {
    "AK": {
        "channels": ["HNZ", "HHZ"],
        "locations": ["", "00"],
        "stations": [
            "CDVT", "DAM1", "DAM2", "FA01", "FA02", "FA05", "FA06", "FA07",
            "FA09", "FA10", "FA12", "FIRE", "GLI", "HIN", "K203", "K204",
            "K205", "K208", "K209", "K210", "K211", "K212", "K213", "K214",
            "K215", "K216", "K217", "K220", "K221", "K222", "K223", "PWL",
        ],
    },
    "AV": {
        "channels": ["HHZ", "HNZ"],
        "locations": ["", "01", "02", "03", "04", "05", "06"],
        "stations": ["DLL", "SDPI"],
    },
    "DE": {
        "channels": ["HHZ", "HNZ"],
        "locations": [""],
        "stations": ["UAF01", "UAF02"],
    },
    "GM": {
        "channels": ["HHZ", "HNZ"],
        "locations": ["", "--", "00", "01", "02", "20"],
        "stations": ["AD02", "AD03", "AD04", "AD06", "AD07", "AD08", "AD09", "AD11", "AD13", "AD14"],
    },
    "IU": {
        "channels": ["HHZ", "HNZ"],
        "locations": ["00", "10", "20", "40"],
        "stations": ["COLA"],
    },
    "NP": {
        "channels": ["HNZ"],
        "locations": [f"{i:02d}" for i in range(1, 33)] + [f"D{i}" for i in range(1, 6)],
        "stations": ["8040", "NIKO"],
    },
    "TA": {
        "channels": ["HNZ"],
        "locations": [""],
        "stations": ["O19K", "P18K", "Q16K", "R17L"],
    },
    "XI": {
        "channels": ["HHZ"],
        "locations": [""],
        "stations": ["APEX1", "APEX2", "APEX4", "APEX8", "APEX9"],
    },
    "XO": {
        "channels": ["HHZ", "HNZ"],
        "locations": [""],
        "stations": [
            "EP14", "EP15", "EP16", "EP21", "EP22", "EP23", "ET17", "ET18",
            "ET19", "ET20", "KD01", "KD02", "KD04", "KD05", "KD12", "KS03",
            "KS11", "KS13", "KT06", "KT07", "KT08", "KT09", "KT10", "WP24",
            "WP25", "WP30", "WS26", "WS27", "WS28",
        ],
    },
    "XV": {
        "channels": ["HHZ"],
        "locations": [""],
        "stations": [
            "F1TN", "F2TN", "F3TN", "F4TN", "F5MN", "F6TP", "F7TV", "F8KN",
            "FAPT", "FNN1", "FNN2", "FPAP", "FTGH",
        ],
    },
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download MiniSEED waveform windows from IRIS for crossing times."
    )
    parser.add_argument(
        "--crossing-file",
        type=str,
        default=DEFAULT_CROSSING_FILE,
        help="Path to the crossing input file.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where MiniSEED files will be saved.",
    )
    parser.add_argument(
        "--time-window",
        type=float,
        default=DEFAULT_TIME_WINDOW,
        help="Half-window length in seconds on each side of the crossing time.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Pause between rows to avoid excessive request rates.",
    )
    return parser.parse_args()


def safe_parse_epoch(value) -> UTCDateTime | None:
    """Convert an epoch-like value into UTCDateTime, else return None."""
    try:
        cleaned_value = float(str(value).replace(",", "").strip())
        if 1e9 < cleaned_value < 2e9:
            return UTCDateTime(cleaned_value)
    except Exception:
        pass
    return None


def read_crossing_file(crossing_file: str) -> pd.DataFrame:
    """Read, label, and clean the crossing file."""
    columns = [
        "Date",
        "FlightNum",
        "X_m",
        "Y_m",
        "Distance_m",
        "ClosestTime",
        "Altitude_m",
        "Speed_mps",
        "Heading",
        "Station",
        "Equipment",
        "Dummy",
    ]

    file_path = Path(crossing_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Crossing file not found: {file_path}")

    data = pd.read_csv(
        file_path,
        sep=",",
        header=None,
        names=columns,
        engine="python",
    )

    data["ClosestTime"] = data["ClosestTime"].apply(safe_parse_epoch)
    data["Station"] = data["Station"].astype(str).str.strip()
    data = data.dropna(subset=["ClosestTime", "Station"])

    return data


def split_workload(data: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Split rows across SLURM array tasks if available."""
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    task_data = data.iloc[task_id::n_tasks]
    return task_data, task_id, n_tasks


def try_download_waveform(
    client: Client,
    network: str,
    station: str,
    location: str,
    channel: str,
    crossing_time: UTCDateTime,
    time_window: float,
):
    """Attempt to download one waveform window from IRIS."""
    try:
        stream = client.get_waveforms(
            network,
            station,
            location,
            channel,
            crossing_time - time_window,
            crossing_time + time_window,
            attach_response=False,
        )
        if stream and len(stream) > 0:
            return stream
    except Exception as exc:
        print(
            f"Download failed for {network}.{station}.{location}.{channel} "
            f"at {crossing_time}: {exc}"
        )
    return None


def download_and_save_waveforms(
    data: pd.DataFrame,
    output_root: Path,
    time_window: float,
    sleep_seconds: float,
) -> dict:
    """Download waveform windows and save them as MiniSEED files."""
    output_root.mkdir(parents=True, exist_ok=True)
    client = Client("IRIS")

    stats = {
        "rows_processed": 0,
        "requests_attempted": 0,
        "files_saved": 0,
        "files_skipped_existing": 0,
        "download_failures": 0,
    }

    for _, row in data.iterrows():
        stats["rows_processed"] += 1

        station = str(row["Station"]).strip()
        crossing_time = row["ClosestTime"]
        time_string = crossing_time.strftime("%Y-%m-%dT%H:%M:%S")

        for network, config in NETWORK_CONFIGS.items():
            if station not in config["stations"]:
                continue

            for channel in config["channels"]:
                for location in config["locations"]:
                    stats["requests_attempted"] += 1

                    stream = try_download_waveform(
                        client=client,
                        network=network,
                        station=station,
                        location=location,
                        channel=channel,
                        crossing_time=crossing_time,
                        time_window=time_window,
                    )

                    if stream is None:
                        stats["download_failures"] += 1
                        continue

                    station_dir = output_root / network / station
                    station_dir.mkdir(parents=True, exist_ok=True)

                    filename = f"{network}.{station}.{location}.{channel}.{time_string}.mseed"
                    output_file = station_dir / filename

                    if output_file.exists():
                        stats["files_skipped_existing"] += 1
                        continue

                    stream.write(str(output_file), format="MSEED")
                    stats["files_saved"] += 1

        time.sleep(sleep_seconds)

    return stats


def main():
    """Run the waveform download workflow."""
    args = parse_arguments()

    if args.time_window <= 0:
        raise ValueError("--time-window must be positive.")

    data = read_crossing_file(args.crossing_file)
    task_data, task_id, n_tasks = split_workload(data)

    print(f"Worker {task_id + 1}/{n_tasks} processing {len(task_data)} rows.")

    stats = download_and_save_waveforms(
        data=task_data,
        output_root=Path(args.output_root),
        time_window=args.time_window,
        sleep_seconds=args.sleep_seconds,
    )

    print("Run complete.")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()