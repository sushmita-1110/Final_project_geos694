"""
Download waveform windows from IRIS for aircraft-station crossings.
"""

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Input crossing table and output directory for downloaded MiniSEED files
CROSSING_FILE = Path("crossings_final50.txt")

OUTPUT_ROOT = Path("output/miniSEED")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TIME_WINDOW = 120
MAX_WORKERS = 4

# Allowed station, channel, and location combinations for each network
# Network configurations 

NETWORK_CONFIGS = {
    "AK": {
        "channels": ["HNZ", "HHZ"],
        "locations": ["", "00"],
        "stations": [
            "CDVT", "DAM1", "DAM2", "FA01", "FA02", "FA05", "FA06",
            "FA07", "FA09", "FA10", "FA12", "FIRE", "GLI", "HIN",
            "K203", "K204", "K205", "K208", "K209", "K210", "K211",
            "K212", "K213", "K214", "K215", "K216", "K217", "K220",
            "K221", "K222", "K223", "PWL",
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
        "stations": [
            "AD02", "AD03", "AD04", "AD06", "AD07",
            "AD08", "AD09", "AD11", "AD13", "AD14",
        ],
    },
    "IU": {
        "channels": ["HHZ", "HNZ"],
        "locations": ["00", "10", "20", "40"],
        "stations": ["COLA"],
    },
    "NP": {
        "channels": ["HNZ"],
        "locations": [f"{i:02d}" for i in range(1, 33)] + 
        [f"D{i}" for i in range(0, 6)],
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
            "EP14", "EP15", "EP16", "EP21", "EP22", "EP23",
            "ET17", "ET18", "ET19", "ET20",
            "KD01", "KD02", "KD04", "KD05", "KD12",
            "KS03", "KS11", "KS13",
            "KT06", "KT07", "KT08", "KT09", "KT10",
            "WP24", "WP25", "WP30",
            "WS26", "WS27", "WS28",
        ],
    },
    "XV": {
        "channels": ["HHZ"],
        "locations": [""],
        "stations": [
            "F1TN", "F2TN", "F3TN", "F4TN",
            "F5MN", "F6TP", "F7TV", "F8KN",
            "FAPT", "FNN1", "FNN2", "FPAP", "FTGH",
        ],
    },
}

# Functions
def safe_parse_time(value):
    """
    Convert a table value to UTCDateTime, returning None if parsing fails.
    """
    try:
        return UTCDateTime(str(value).strip())
    except Exception:
        return None


def download_with_iris(client, network, station, location, channel, timestamp, outdir):
    """
    Download one waveform window from IRIS and save it as MiniSEED.
    """
    start = timestamp - TIME_WINDOW
    end = timestamp + TIME_WINDOW

    try:
        st = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=start,
            endtime=end,
            attach_response=False,
        )
    except Exception:
        return 0

    if not st:
        return 0

    outdir.mkdir(parents=True, exist_ok=True)

    loc_tag = location if location else "NONE"
    time_tag = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    outfile = outdir / f"{network}.{station}.{loc_tag}.{channel}_{time_tag}.mseed"

    try:
        st.write(str(outfile), format="MSEED")
        return 1
    except Exception:
        return 0


def process_row(row, network, channels, locations):
    """
    Try valid channel/location combinations for one crossing-table row.
    """
    client = Client("IRIS")

    station = str(row["station"]).strip()
    timestamp = row["time_str"]
    row_channel = str(row["channel"]).strip()
    row_location = str(row["location"]).strip()
    equipment = str(row["equipment"]).strip()

    time_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"\n{network}.{station} at {time_str}  equipment={equipment}")

    channel_list = [row_channel] if row_channel else channels
    location_list = [row_location] if row_location in locations else locations

    saved = 0

    for channel in channel_list:
        found = False

        for location in location_list:
            loc_display = location if location else "None"
            print(f"  -> Trying loc={loc_display}, chan={channel}")

            outdir = OUTPUT_ROOT / network / station

            count = download_with_iris(
                client,
                network,
                station,
                location,
                channel,
                timestamp,
                outdir,
            )

            if count:
                print(f"     Saved: {count} file(s) to {outdir}")
                saved += count
                found = True
                break

        if not found:
            print(f"     No data found for channel {channel}")

    time.sleep(0.1)
    return saved


def main():
    # Read and clean the crossing table
    df = pd.read_csv(CROSSING_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    df["time_str"] = df["time_str"].apply(safe_parse_time)
    df["network"] = df["network"].astype(str).str.strip()
    df["station"] = df["station"].astype(str).str.strip()
    df["location"] = df["location"].fillna("").astype(str).str.strip()
    df["channel"] = df["channel"].fillna("").astype(str).str.strip()
    df["equipment"] = df["equipment"].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["time_str", "station", "network"])

    print(df.columns.tolist())
    print(f"Loaded {len(df)} total timestamps.")

    total_saved = 0

    # Loop through each configured network and process matching rows
    for network, config in NETWORK_CONFIGS.items():
        print(f"Processing Network: {network}")

        stations = config["stations"]
        channels = config["channels"]
        locations = config["locations"]

        net_df = df[(df["network"] == network) & (df["station"].isin(stations))]
        print(f"Found {len(net_df)} timestamps for this network.")

        if net_df.empty:
            continue

        # Download rows in parallel for faster waveform retrieval
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_row, row, network, channels, locations)
                for _, row in net_df.iterrows()
                ]
            
            for future in as_completed(futures):
                try:
                    total_saved += future.result()
                except Exception as e:
                    print(f"Worker failed: {e}")


    print(f"TOTAL MiniSEED files saved: {total_saved}")
    print(f"Output directory: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()