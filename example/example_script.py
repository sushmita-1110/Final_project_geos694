from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import spectrogram

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
sys.path.append(str(REPO_DIR))

from flight_query import FlightVizPDF

# Paths
FLIGHT_DIR = BASE_DIR / "flightradar24"
STATION_FILE = REPO_DIR / "gmap-stations_H??.txt"
CROSSING_FILE = REPO_DIR / "crossings_final50.txt"
OUTPUT_DIR = BASE_DIR / "C130_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# One-flight example
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def base_name():
    time_tag = T0.strftime("%Y-%m-%dT%H-%M-%S")
    return f"{NETWORK}.{STATION}.{LOCATION}.{CHANNEL}_{EQUIPMENT}_{time_tag}"


def remove_median(spec):
    return np.clip(spec - np.median(spec, axis=1, keepdims=True), 0, None)


def download_waveform():
    client = Client("IRIS")
    st = client.get_waveforms(
        network=NETWORK,
        station=STATION,
        location=LOCATION,
        channel=CHANNEL,
        starttime=T0 - WINDOW_SEC,
        endtime=T0 + WINDOW_SEC,
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
            f"starting {T0.strftime('%Y-%m-%dT%H:%M:%S')} "
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
        ax4.tick_params(bottom=False, labelbottom=False, left=True, labelleft=True)
        ax4.grid(axis="y", alpha=0.3)

        png = OUTPUT_DIR / f"{base_name()}_spectrogram.png"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        return png

    finally:
        if fig is not None:
            plt.close(fig)


def make_flight_query_style_pdf():
    viz = FlightVizPDF(
        flight_dir=FLIGHT_DIR,
        station_file=STATION_FILE,
        crossing_file=CROSSING_FILE,
        utm_zone=6,
    )

    date = T0.strftime("%Y%m%d")
    results = viz.crossings[
        (viz.crossings["network"].astype(str).str.strip().str.upper() == NETWORK)
        & (viz.crossings["station"].astype(str).str.strip().str.upper() == STATION)
        & (viz.crossings["flight_num"].astype(str) == FLIGHT_NUM)
        & (viz.crossings["date"].astype(str) == date)
        & (viz.crossings["channel"].astype(str).str.strip().str.upper() == CHANNEL)
    ].copy()

    if results.empty:
        raise ValueError("No matching crossing found in crossings_final50.txt")

    viz.generate_grouped_pdfs(
        results,
        output_dir=OUTPUT_DIR,
        plot_all_stations=False,
        suffix_func=lambda group: f"station_{group.iloc[0]['station']}",
    )

    row = results.iloc[0]
    return OUTPUT_DIR / (
        f"{row['date']}_{row['flight_num']}_{row['equipment']}"
        f"_station_{row['station']}.pdf"
    )


def main():
    print("Running C130 workflow")
    print(f"Flight: {FLIGHT_NUM}")
    print(f"Station: {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}")
    print(f"Time: {T0}")

    st, mseed_path = download_waveform()
    png_path = make_spectrogram_png(st)
    pdf_path = make_flight_query_style_pdf()

    print("\nFinished")
    print(f"Waveform saved to: {mseed_path}")
    print(f"Spectrogram saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")


if __name__ == "__main__":
    main()