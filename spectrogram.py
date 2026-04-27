"""
Make waveform and spectrogram figures from downloaded MiniSEED files.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from matplotlib.gridspec import GridSpec
from obspy import UTCDateTime, read

# Input crossing table and output directory for spectrogram figures
SUMMARY_CSV = Path("crossings_final50.txt")
OUTPUT_ROOT = Path("output/spectrogram")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 120
WIN_LEN = 1

# Highpass filter frequency in Hz
HP_FREQ = 10.0

def parse_time(x):
    """ Convert an input value to ObsPy UTCDateTime. """
    try: 
        return UTCDateTime(str(x).strip())
    except Exception: 
        return None

def remove_median(S):
    """ Remove the frequency-wise median """
    return np.clip(S - np.median(S, axis=1, keepdims=True), 0, None)

def make_spectrogram(mseed_file, t0, aircraft, net, sta, cha, loc, outdir, rank=0):
    """
    Create and save one waveform-plus-spectrogram figure.
    """
    fig = None
    try:
        st = read(str(mseed_file))
        if not st: 
            return None
        st.merge(method=1, fill_value=0)
        tr = st[0]
        if not (tr.stats.starttime <= t0 <= tr.stats.endtime): 
            return None
        tr.trim(t0 - WINDOW_SEC, t0 + WINDOW_SEC, pad=True, fill_value=0)

        raw = tr.data.astype(float)
        fs = float(tr.stats.sampling_rate)
        if fs / 2 <= HP_FREQ: 
            return None

        trf = tr.copy()
        trf.detrend("demean")
        trf.taper(max_percentage=0.02, type="cosine")
        trf.filter("highpass", freq=HP_FREQ, corners=4, zerophase=True)
        wf, tw = trf.data.astype(float), trf.times()

        nper = max(int(WIN_LEN * fs), 8)
        if len(raw) < 2 * nper: 
            return None
        f, t, Sxx = spectrogram(raw, fs, scaling="density", nperseg=nper,
                                noverlap=int(0.9 * nper), detrend="constant")
        if Sxx.shape[1] < 2: 
            return None

        spec = 10 * np.log10(remove_median(Sxx) + 1e-12)
        mask = f >= HP_FREQ
        if not np.any(mask): 
            return None

        spec2 = spec[mask]
        finite_all = spec2[np.isfinite(spec2)]
        finite_mid = spec2[:, len(t)//2]
        finite_mid = finite_mid[np.isfinite(finite_mid)]
        if finite_all.size == 0 or finite_mid.size == 0: 
            return None

        vmin = np.percentile(finite_all, 65)
        vmax = np.max(finite_mid)
        if vmax <= vmin: 
            vmax = vmin + 1

        side = 10 * np.log10(np.median(Sxx[mask], axis=1) + 1e-12)
        fside = f[mask]

        title = f"{net}.{sta}.{loc}.{cha} {aircraft} - starting {t0.strftime('%Y-%m-%dT%H:%M:%S')}  [Waveform HP {HP_FREQ:.0f} Hz]"

        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(2, 3, figure=fig,
                      height_ratios=[1, 1.1],
                      width_ratios=[0.10, 1, 0.03],   # thinner left figure
                      hspace=0.22, wspace=0.06)

        ax_blank = fig.add_subplot(gs[0, 0]); 
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
            if lim > 0: ax1.set_ylim(-2.5 * lim, 2.5 * lim)

        im = ax2.pcolormesh(t, f, spec, shading="gouraud", 
                            cmap="pink_r", vmin=vmin, vmax=vmax)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("")
        ax2.tick_params(left=False, labelleft=False)
        ax2.set_xlim(0, 2 * WINDOW_SEC)
        ax2.set_ylim(0, int(fs / 2))

        # Colourbar
        cb = plt.colorbar(im, cax=ax3)
        cb.set_label("Relative Amplitude (dB)")
 
        # Side spectrum panel
        ax4.plot(side, fside, color="#ff7f00", lw=1.5)
        ax4.set_ylim(0, int(fs / 2))
        ax4.invert_xaxis()
        ax4.set_ylabel("Frequency (Hz)", labelpad=-2)   # closer to axis
        ax4.yaxis.set_label_position("left")
        ax4.yaxis.tick_left()
        ax4.tick_params(bottom=False, labelbottom=False, left=True, labelleft=True)
        ax4.grid(axis="y", alpha=0.3)

        # Save output PNG
        outdir.mkdir(parents=True, exist_ok=True)
        safe_time = t0.strftime("%Y-%m-%dT%H-%M-%S")
        aircraft = str(aircraft).strip().replace("/", "_").replace(" ", "_")
        png = outdir / f"{int(rank)}_{net}.{sta}.{loc}.{cha}_{aircraft}_{safe_time}.png"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        return png

    except Exception as e:
        print("Error:", e)
        return None
    finally:
        if fig is not None:
            plt.close(fig)

def main():
    """
    Read the crossing table, match waveform files, and generate figures.
    """
    df = pd.read_csv(SUMMARY_CSV, sep="\t")
    df.columns = df.columns.str.strip()

    df["time_str"] = df["time_str"].apply(parse_time)
    df["network"] = df["network"].astype(str).str.strip()
    df["station"] = df["station"].astype(str).str.strip()
    df["channel"] = df["channel"].fillna("").astype(str).str.strip()
    df["location"] = df["location"].fillna("").astype(str).str.strip()
    df["equipment"] = df["equipment"].fillna("").astype(str).str.strip()
    df["d0_m"] = pd.to_numeric(df["d0_m"], errors="coerce")
    df = df.dropna(subset=["time_str", "network", "station", "channel"]).copy()

    # Rebuild the expected MiniSEED file path from the fetch-data workflow
    def build_outfile(r):
        loc_tag = r["location"] if r["location"] else "NONE"
        time_tag = r["time_str"].strftime("%Y-%m-%dT%H-%M-%S")
        filename = (
            f'{r["network"]}.{r["station"]}.{loc_tag}.'
            f'{r["channel"]}_{time_tag}.mseed'
        )
        return Path("output/miniSEED") / r["network"] / r["station"] / filename
    
    df["outfile"] = df.apply(build_outfile, axis=1)
    df = df[df["outfile"].apply(lambda p: p.exists())].copy()


    # Rank rows by distance (d0) rank within each station/channel/location group
    df = df.sort_values(["network", "station", "channel", "location", "d0_m"])
    df["d0_rank"] = df.groupby(["network", "station", 
                                "channel", "location"]).cumcount() + 1

    ok = fail = 0

    # Build one figure for each waveform file that exists
    for _, r in df.iterrows():
        loc = r["location"] if r["location"] else "NONE"
        made = make_spectrogram(
            r["outfile"], r["time_str"], r["equipment"],
            r["network"], r["station"], r["channel"], loc,
            OUTPUT_ROOT / r["network"] / r["station"] / r["channel"] / loc,
            r["d0_rank"],
        )

        if made:
            ok += 1
        else:
            fail += 1

    print(f"Spectrograms created: {ok}")
    print(f"Rows failed: {fail}")


if __name__ == "__main__":
    main()    