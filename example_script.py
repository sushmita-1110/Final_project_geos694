from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from matplotlib.backends.backend_pdf import PdfPages
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import spectrogram


# ---------------------------------------------------------------------
# One-flight example settings
# ---------------------------------------------------------------------
FLIGHT_DIR = Path("flightradar24")
STATION_FILE = Path(
    "gmap-stations_H??.txt"
)
OUTPUT_DIR = Path("C130_output")

NETWORK = "AK"
STATION = "FIRE"
LOCATION = ""
CHANNEL = "HNZ"
EQUIPMENT = "C130"
FLIGHT_NUM = "527808805"
T0 = UTCDateTime("2019-02-11T00:45:50")

WINDOW_SEC = 120
WIN_LEN = 1.0
HP_FREQ = 10.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def base_name():
    time_tag = T0.strftime("%Y-%m-%dT%H-%M-%S")
    return f"{NETWORK}.{STATION}.{LOCATION}.{CHANNEL}_{EQUIPMENT}_{time_tag}"


def load_station():
    stations = pd.read_csv(STATION_FILE, sep="|")
    stations["Station"] = stations["Station"].astype(str).str.strip().str.upper()
    row = stations[stations["Station"] == STATION]
    if row.empty:
        raise ValueError(f"Station '{STATION}' not found in {STATION_FILE}")
    return row.iloc[0]


def load_flight_track():
    date = T0.strftime("%Y%m%d")
    path = FLIGHT_DIR / f"{date}_positions" / f"{date}_{FLIGHT_NUM}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Flight track file not found: {path}")
    return pd.read_csv(path)


def load_flight_metadata():
    date = T0.strftime("%Y%m%d")
    path = FLIGHT_DIR / f"{date}_flights.csv"
    info = {"callsign": "Unknown", "tail": "Unknown", "equip": EQUIPMENT}

    if not path.exists():
        return info

    df = pd.read_csv(path)
    if "flight_id" not in df.columns:
        return info

    df["flight_id"] = df["flight_id"].astype(str)
    row = df[df["flight_id"] == FLIGHT_NUM]
    if row.empty:
        return info

    row = row.iloc[0]
    info["callsign"] = (
        str(row.get("callsign", "Unknown"))
        if pd.notna(row.get("callsign"))
        else "Unknown"
    )
    info["tail"] = (
        str(row.get("aircraft_id", "Unknown"))
        if pd.notna(row.get("aircraft_id"))
        else "Unknown"
    )
    info["equip"] = (
        str(row.get("equip", EQUIPMENT))
        if pd.notna(row.get("equip"))
        else EQUIPMENT
    )
    return info


def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, np.asarray(lon2), np.asarray(lat2)]
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def cumulative_distance_km(lon, lat, zone=6):
    proj = pyproj.Proj(proj="utm", zone=zone, ellps="WGS84")
    x, y = proj(np.asarray(lon), np.asarray(lat))
    out = np.zeros(len(x))
    out[1:] = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)) / 1000.0
    return out


def remove_median(spec):
    return np.clip(spec - np.median(spec, axis=1, keepdims=True), 0, None)


# ---------------------------------------------------------------------
# Step 1: Download waveform from IRIS
# ---------------------------------------------------------------------
def download_waveform():
    client = Client("IRIS")
    start = T0 - WINDOW_SEC
    end = T0 + WINDOW_SEC

    st = client.get_waveforms(
        network=NETWORK,
        station=STATION,
        location=LOCATION,
        channel=CHANNEL,
        starttime=start,
        endtime=end,
        attach_response=False,
    )
    st.merge(method=1, fill_value=0)

    out = OUTPUT_DIR / f"{base_name()}.mseed"
    st.write(str(out), format="MSEED")
    return st, out


# ---------------------------------------------------------------------
# Step 2: Make spectrogram PNG using the same logic as spectrogram.py
# ---------------------------------------------------------------------
def make_spectrogram_png(st):
    fig = None
    try:
        tr = st[0].copy()
        fs = float(tr.stats.sampling_rate)

        raw = tr.data.astype(float)
        if fs / 2 <= HP_FREQ:
            return None

        trf = tr.copy()
        trf.detrend("demean")
        trf.taper(max_percentage=0.02, type="cosine")
        trf.filter("highpass", freq=HP_FREQ, corners=4, zerophase=True)

        wf = trf.data.astype(float)
        tw = trf.times()

        nper = max(int(WIN_LEN * fs), 8)
        if len(raw) < 2 * nper:
            return None

        f, t, sxx = spectrogram(
            raw,
            fs,
            scaling="density",
            nperseg=nper,
            noverlap=int(0.9 * nper),
            detrend="constant",
        )

        if sxx.shape[1] < 2:
            return None

        spec = 10 * np.log10(remove_median(sxx) + 1e-12)

        mask = f >= HP_FREQ
        if not np.any(mask):
            return None

        spec2 = spec[mask]
        f2 = f[mask]

        finite_all = spec2[np.isfinite(spec2)]
        finite_mid = spec2[:, len(t) // 2]
        finite_mid = finite_mid[np.isfinite(finite_mid)]
        if finite_all.size == 0 or finite_mid.size == 0:
            return None

        vmin = np.percentile(finite_all, 65)
        vmax = np.max(finite_mid)
        if vmax <= vmin:
            vmax = vmin + 1

        side = 10 * np.log10(np.median(sxx[mask], axis=1) + 1e-12)

        title = (
            f"{NETWORK}.{STATION}.{LOCATION}.{CHANNEL} {EQUIPMENT} - "
            f"starting {T0.strftime('%Y-%m-%dT%H:%M:%S')}  "
            f"[Waveform HP {HP_FREQ:.0f} Hz]"
        )

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[1, 1.1],
            width_ratios=[0.10, 1, 0.03],
            hspace=0.22,
            wspace=0.06,
        )

        ax_blank = fig.add_subplot(gs[0, 0])
        ax_blank.axis("off")

        ax1 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[1, 2])

        ax1.plot(tw, wf, "k", lw=0.5)
        ax1.set_title(title)
        ax1.set_ylabel("Counts")
        ax1.set_xlim(0, 2 * WINDOW_SEC)

        good = wf[np.isfinite(wf)]
        if good.size:
            lim = np.percentile(np.abs(good), 99.5)
            if lim > 0:
                ax1.set_ylim(-2.5 * lim, 2.5 * lim)

        im = ax2.pcolormesh(
            t,
            f2,
            spec2,
            shading="gouraud",
            cmap="pink_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("")
        ax2.tick_params(left=False, labelleft=False)
        ax2.set_xlim(0, 2 * WINDOW_SEC)
        ax2.set_ylim(0, int(fs / 2))

        cb = plt.colorbar(im, cax=ax3)
        cb.set_label("Relative Amplitude (dB)")

        ax4.plot(side, f2, color="#ff7f00", lw=1.5)
        ax4.set_ylim(0, int(fs / 2))
        ax4.invert_xaxis()
        ax4.set_ylabel("Frequency (Hz)", labelpad=-2)
        ax4.yaxis.set_label_position("left")
        ax4.yaxis.tick_left()
        ax4.tick_params(bottom=False, labelbottom=False, left=True, 
                        labelleft=True)
        ax4.grid(axis="y", alpha=0.3)

        png = OUTPUT_DIR / f"{base_name()}_spectrogram.png"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        return png

    finally:
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------
# Step 3: Make simple flight-summary PDF
# ---------------------------------------------------------------------
def make_pdf_summary(flight, station, info):
    lon = flight["longitude"].to_numpy()
    lat = flight["latitude"].to_numpy()
    alt_m = flight["altitude"].to_numpy() * 0.3048

    d_km = haversine_distance(
        station["Longitude"],
        station["Latitude"],
        lon,
        lat,
    )
    idx = int(np.argmin(d_km))
    dist_track_km = cumulative_distance_km(lon, lat)

    pdf_path = OUTPUT_DIR / f"{base_name()}_flightpath.pdf"

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.30, 
                              wspace=0.25)

        ax_info = fig.add_subplot(gs[0, 0])
        ax_map = fig.add_subplot(gs[0, 1])
        ax_profile = fig.add_subplot(gs[1, :])

        ax_info.axis("off")
        lines = [
            f"Flight number: {FLIGHT_NUM}",
            f"Aircraft type: {info['equip']}",
            f"Callsign: {info['callsign']}",
            f"Tail: {info['tail']}",
            f"Network: {NETWORK}",
            f"Station: {STATION}",
            f"Channel: {CHANNEL}",
            f"Crossing time (UTC): {T0.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Closest flight-track distance: {d_km[idx] * 1000:.0f} m",
            f"Altitude at closest point: {alt_m[idx]:.0f} m",
        ]
        y = 0.95
        for line in lines:
            ax_info.text(0.02, y, line, fontsize=10, va="top")
            y -= 0.09

        ax_map.plot(lon, lat, "k-", lw=1.2, label="Flight track")
        ax_map.scatter(lon[0], lat[0], s=60, c="green", label="Start", zorder=3)
        ax_map.scatter(lon[-1], lat[-1], s=60, c="red", label="End", zorder=3)
        ax_map.scatter(
            station["Longitude"],
            station["Latitude"],
            s=90,
            marker="v",
            c="blue",
            label=STATION,
            zorder=4,
        )
        ax_map.scatter(
            lon[idx],
            lat[idx],
            s=80,
            c="orange",
            label="Closest approach",
            zorder=4,
        )
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        ax_map.set_title("Flight path and station")
        ax_map.grid(True, alpha=0.3)
        ax_map.legend(fontsize=8)

        ax_profile.plot(dist_track_km, alt_m, "k-", lw=1.2)
        ax_profile.scatter(
            dist_track_km[0], alt_m[0], s=50, c="green", zorder=3, label="Start"
        )
        ax_profile.scatter(
            dist_track_km[-1], alt_m[-1], s=50, c="red", zorder=3, label="End"
        )
        ax_profile.axvline(
            dist_track_km[idx],
            color="orange",
            ls="--",
            lw=1.5,
            label="Closest approach",
        )
        ax_profile.scatter(
            dist_track_km[idx], alt_m[idx], s=70, c="orange", zorder=4
        )
        ax_profile.set_xlabel("Cumulative horizontal distance (km)")
        ax_profile.set_ylabel("Altitude (m)")
        ax_profile.set_title("Flight altitude profile")
        ax_profile.grid(True, alpha=0.3)
        ax_profile.legend(fontsize=8)

        fig.suptitle(base_name(), fontsize=13, y=0.98)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return pdf_path


def main():
    print("Running C130-flight example")
    print(f"Flight: {FLIGHT_NUM}")
    print(f"Station: {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}")
    print(f"Time: {T0}")

    station = load_station()
    flight = load_flight_track()
    info = load_flight_metadata()

    st, mseed_path = download_waveform()
    png_path = make_spectrogram_png(st)
    pdf_path = make_pdf_summary(flight, station, info)

    print(f"Waveform saved to: {mseed_path}")
    print(f"Spectrogram saved to: {png_path}")
    print(f"Flight path saved to: {pdf_path}")


if __name__ == "__main__":
    main()