from pathlib import Path
import math

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pyproj


FLIGHT_DIR = Path("/scratch/irseppi/nodal_data/flightradar24")
STATION_FILE = Path(
    "/home/smaurya/repository/parkshwynodal/local_data_fetch/gmap-stations_H??.txt"
)
CROSSING_FILE = Path(
    "/home/smaurya/repository/parkshwynodal/local_data_fetch/crossings_final50.txt"
)


class FlightVizPDF:
    """Query aircraft crossings and generate flight-path PDF."""

    def __init__(self, flight_dir, station_file, crossing_file, utm_zone=6):
        self.flight_dir = Path(flight_dir)
        self.utm = pyproj.Proj(proj="utm", zone=utm_zone, ellps="WGS84")
        self.stations = self._load_stations(station_file)
        self.crossings = self._load_crossings(crossing_file)
        print(f"Loaded {len(self.crossings)} crossings and {len(self.stations)} stations")

    def _load_stations(self, station_file):
        stations = pd.read_csv(station_file, sep="|")
        stations["UTM_X"], stations["UTM_Y"] = zip(*[
            self.utm(lon, lat)
            for lat, lon in zip(stations["Latitude"], stations["Longitude"])
        ])
        return stations

    @staticmethod
    def _load_crossings(crossing_file):
        df = pd.read_csv(crossing_file, sep="\t")
        df.columns = df.columns.str.strip()
        df["station"] = df["station"].astype(str).str.strip().str.upper()
        df["flight_num"] = (
            df["flight_num"].astype(str).str.strip().str.split(".").str[0]
        )
        df["equipment"] = df["equipment"].astype(str).str.strip()
        df["date"] = pd.to_datetime(df["time_str"]).dt.strftime("%Y%m%d")

        for col in ["distance_m", "altitude_m_x", "d0_m", "heading"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _read_csv(path, **kwargs):
        path = Path(path)
        if not path.exists():
            return None
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as exc:
            print(f"Could not read {path}: {exc}")
            return None

    @staticmethod
    def _filter_crossings(df, start_date=None, end_date=None, max_crossing_km=None):
        if start_date:
            end_date = end_date or start_date
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if max_crossing_km is not None:
            df = df[df["distance_m"] <= max_crossing_km * 1000]
        return df

    @staticmethod
    def haversine_distance(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))

    @staticmethod
    def cumulative_distance(x, y, z=None, mode="horizontal"):
        x = np.asarray(x)
        y = np.asarray(y)
        if mode == "horizontal":
            step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        elif mode == "true3d" and z is not None:
            z = np.asarray(z)
            step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        else:
            raise ValueError("mode must be 'horizontal' or 'true3d'")
        out = np.zeros(len(x))
        out[1:] = np.cumsum(step)
        return out / 1000.0

    @staticmethod
    def moving_mean(values, window=5):
        out = np.copy(values)
        for i in range(len(values)):
            j0 = max(0, i - window // 2)
            j1 = min(len(values), i + window // 2 + 1)
            out[i] = np.mean(values[j0:j1])
        return out

    def _flight_path(self, date, flight_num):
        return self.flight_dir / f"{date}_positions" / f"{date}_{flight_num}.csv"

    def load_flights_metadata(self, date):
        df = self._read_csv(self.flight_dir / f"{date}_flights.csv")
        if df is not None and "flight_id" in df.columns:
            df["flight_id"] = df["flight_id"].astype(str)
        return df

    def get_aircraft_info(self, date, flight_num):
        df = self.load_flights_metadata(date)
        default = {"tail": "Unknown", "callsign": "Unknown", "equip": "Unknown"}

        if df is None or "flight_id" not in df.columns:
            return default

        match = df[df["flight_id"] == str(flight_num)]
        if match.empty:
            return default

        row = match.iloc[0]
        return {
            "tail": (
                row["aircraft_id"]
                if pd.notna(row.get("aircraft_id"))
                and str(row["aircraft_id"]).strip()
                else "Unknown"
            ),
            "callsign": (
                row["callsign"]
                if pd.notna(row.get("callsign"))
                and str(row["callsign"]).strip()
                else "Unknown"
            ),
            "equip": (
                row["equip"]
                if pd.notna(row.get("equip"))
                and str(row["equip"]).strip()
                else "Unknown"
            ),
        }

    def query_flights_by_numbers(
        self, flight_numbers, start_date=None, end_date=None, max_crossing_km=None
    ):
        df = self.crossings[
            self.crossings["flight_num"].isin([str(f) for f in flight_numbers])
        ].copy()
        df = self._filter_crossings(df, start_date, end_date, max_crossing_km)
        print("No flights found" if df.empty else f"{len(df)} crossings")
        return df

    def query_flights_by_station(
        self,
        station_name,
        radius_km=50,
        start_date=None,
        end_date=None,
        max_crossing_km=None,
    ):
        match = self.stations[self.stations["Station"] == station_name]
        if match.empty:
            print("Station not found")
            return pd.DataFrame()

        station_lon = match.iloc[0]["Longitude"]
        station_lat = match.iloc[0]["Latitude"]
        df = self._filter_crossings(
            self.crossings.copy(), start_date, end_date, max_crossing_km
        )
        keep = []

        for (date, flight_num), _ in df.groupby(["date", "flight_num"]):
            flight = self._read_csv(self._flight_path(date, flight_num))
            if flight is None:
                continue
            try:
                d = self.haversine_distance(
                    station_lon,
                    station_lat,
                    flight["longitude"].values,
                    flight["latitude"].values,
                )
                if (d <= radius_km).any():
                    keep.append((date, flight_num))
            except Exception:
                continue

        if not keep:
            print("No flights found")
            return pd.DataFrame()

        out = pd.concat(
            [df[(df["date"] == d) & (df["flight_num"] == f)] for d, f in keep],
            ignore_index=True,
        )
        print(f"{len(out)} crossings")
        return out

    def query_flight_station_crossing(self, flight_num, station_name, date):
        df = self.crossings[
            (self.crossings["flight_num"] == str(flight_num))
            & (self.crossings["station"] == str(station_name).upper())
            & (self.crossings["date"] == str(date))
        ]

        if df.empty:
            print("No crossing found")
            return df

        row = df.iloc[0]
        print(f"Found: {row['distance_m']:.0f} m, {row['time_str']}")
        return df

    def summarize_query_results(self, df):
        if not df.empty:
            print(f"{len(df)} crossings")

    def _geom(self, flight):
        start_lon = flight["longitude"].iloc[0]
        start_lat = flight["latitude"].iloc[0]
        end_lon = flight["longitude"].iloc[-1]
        end_lat = flight["latitude"].iloc[-1]

        fx, fy = zip(*[
            self.utm(lon, lat)
            for lat, lon in zip(flight["latitude"], flight["longitude"])
        ])
        fx, fy = np.asarray(fx), np.asarray(fy)
        fz = flight["altitude"].values * 0.3048

        lon_min, lon_max = flight["longitude"].min(), flight["longitude"].max()
        lat_min, lat_max = flight["latitude"].min(), flight["latitude"].max()
        lon_pad = max((lon_max - lon_min) * 0.1, 0.1)
        lat_pad = max((lat_max - lat_min) * 0.1, 0.1)

        return {
            "fx": fx,
            "fy": fy,
            "fz": fz,
            "dist_h": self.cumulative_distance(fx, fy, mode="horizontal"),
            "dist_3d": self.cumulative_distance(fx, fy, fz, mode="true3d"),
            "bbox": {
                "lon_min": lon_min - lon_pad,
                "lon_max": lon_max + lon_pad,
                "lat_min": lat_min - lat_pad,
                "lat_max": lat_max + lat_pad,
            },
            "start_lon": start_lon,
            "start_lat": start_lat,
            "end_lon": end_lon,
            "end_lat": end_lat,
        }

    def _speed(self, flight, geom):
        fx, fy, fz, dist_h = (
            geom["fx"],
            geom["fy"],
            geom["fz"],
            geom["dist_h"],
        )

        if "snapshot_id" not in flight.columns:
            d = np.abs(np.diff(fz))
            d = np.append(d, d[-1])
            speeds = 40 + 20 * (d / (np.max(d) + 1))
            return np.clip(speeds, 10, None), np.max(speeds)

        try:
            t = pd.to_datetime(flight["snapshot_id"], unit="s")
            dt = np.diff(t.values).astype("timedelta64[s]").astype(float)
            ds = np.sqrt(np.diff(fx) ** 2 + np.diff(fy) ** 2 + np.diff(fz) ** 2)

            speeds = np.zeros(len(fx))
            valid = dt > 0
            speeds[:-1][valid] = ds[valid] / dt[valid]
            speeds[-1] = speeds[-2] if len(speeds) > 1 else 0.0
            speeds = self.moving_mean(speeds, 5)

            max_speed = np.nanmax(speeds) if np.any(speeds > 0) else 50.0
            if np.std(speeds) < 1.0 or max_speed < 1.0:
                total_t = (t.iloc[-1] - t.iloc[0]).total_seconds()
                if total_t > 0:
                    avg_speed = (dist_h[-1] * 1000.0) / total_t
                    d = np.abs(np.diff(fz))
                    d = np.append(d, d[-1])
                    speeds = avg_speed * (1 + 0.3 * (d / np.max(d + 1e-10)))
                    max_speed = np.max(speeds)
                else:
                    speeds = np.linspace(30, 60, len(fx))
                    max_speed = 60.0

            return speeds, max_speed
        except Exception as exc:
            print(f"Speed calculation failed: {exc}")
            return np.linspace(30, 60, len(fx)), 60.0

    def _station_records(self, crosses, bbox, plot_all_stations):
        out, seen = [], set()

        def add(name, lon, lat, is_crossing):
            if name not in seen:
                seen.add(name)
                out.append(
                    {"name": name, "lon": lon, "lat": lat, "is_crossing": is_crossing}
                )

        for _, crossing in crosses.iterrows():
            match = self.stations[self.stations["Station"] == crossing["station"]]
            if not match.empty:
                add(
                    crossing["station"],
                    match.iloc[0]["Longitude"],
                    match.iloc[0]["Latitude"],
                    True,
                )

        if plot_all_stations:
            for _, st in self.stations.iterrows():
                lon, lat = st["Longitude"], st["Latitude"]
                if (
                    bbox["lon_min"] <= lon <= bbox["lon_max"]
                    and bbox["lat_min"] <= lat <= bbox["lat_max"]
                ):
                    add(st["Station"], lon, lat, False)

        return out

    @staticmethod
    def _time_fields(flight):
        if "snapshot_id" not in flight.columns:
            return "N/A", "N/A", "N/A"
        start = pd.to_datetime(flight["snapshot_id"].iloc[0], unit="s")
        end = pd.to_datetime(flight["snapshot_id"].iloc[-1], unit="s")
        return (
            start.strftime("%H:%M:%S"),
            end.strftime("%H:%M:%S"),
            f"{(end - start).total_seconds() / 60.0:.1f} min",
        )

    def _draw_info(self, fig, flight_num, flight, info, geom, max_speed):
        ax = fig.add_axes([0.05, 0.55, 0.22, 0.3])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        start_str, end_str, duration_str = self._time_fields(flight)
        sx, sy = self.utm(geom["start_lon"], geom["start_lat"])
        ex, ey = self.utm(geom["end_lon"], geom["end_lat"])
        straight_dist = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2) / 1000.0

        lines = [
            f"Flight: {flight_num}",
            f"Callsign: {info['callsign']}",
            f"Tail: {info['tail']}",
            f"Type: {info['equip']}",
            f"Start: {start_str}",
            f"End: {end_str}",
            f"Duration: {duration_str}",
            f"Max altitude: {geom['fz'].max():.0f} m",
            f"Max speed: {max_speed:.1f} m/s",
            f"Start to end distance: {straight_dist:.1f} km",
            f"Total horizontal distance: {geom['dist_h'][-1]:.1f} km",
            f"Total 3D distance: {geom['dist_3d'][-1]:.1f} km",
        ]
        y = 0.92
        for line in lines:
            ax.text(0.08, y, line, fontsize=9, va="top")
            y -= 0.08

    @staticmethod
    def _marker(ax, x, y, label, face, **kwargs):
        ax.text(
            x,
            y,
            label,
            fontsize=10 if label in {"S", "E"} else 6,
            fontweight="bold",
            color="white",
            va="center",
            ha="center",
            bbox=dict(
                boxstyle="circle,pad=0.3" if label in {"S", "E"} else "circle,pad=0.2",
                facecolor=face,
                edgecolor="black",
                linewidth=2 if label in {"S", "E"} else 1,
            ),
            **kwargs,
        )

    @staticmethod
    def _scale_bar(ax, bbox, use_cartopy=False):
        lon_range = bbox["lon_max"] - bbox["lon_min"]
        lat_avg = (bbox["lat_min"] + bbox["lat_max"]) / 2.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(lat_avg))
        width_m = lon_range * meters_per_deg_lon
        lengths = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        length_m = min(lengths, key=lambda x: abs(x - width_m * 0.2))
        length_deg = length_m / meters_per_deg_lon
        tick_h = (bbox["lat_max"] - bbox["lat_min"]) * 0.015
        label = f"{length_m / 1000:.0f} km" if length_m >= 1000 else f"{length_m:.0f} m"

        if use_cartopy:
            x0 = math.ceil(bbox["lon_min"])
            y0 = bbox["lat_min"] + (bbox["lat_max"] - bbox["lat_min"]) * 0.08
            plot_kwargs = {"transform": ccrs.PlateCarree(), "zorder": 20}
            text_kwargs = {"transform": ccrs.PlateCarree(), "zorder": 21}
        else:
            x0 = bbox["lon_min"] + lon_range * 0.05
            y0 = bbox["lat_min"] + (bbox["lat_max"] - bbox["lat_min"]) * 0.08
            plot_kwargs = {"zorder": 20}
            text_kwargs = {"zorder": 21}

        ax.plot([x0, x0 + length_deg], [y0, y0], "k-", linewidth=1.5, **plot_kwargs)
        ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], "k-", linewidth=1, **plot_kwargs)
        ax.plot([x0 + length_deg, x0 + length_deg], [y0 - tick_h, y0 + tick_h], "k-", linewidth=1, **plot_kwargs)
        ax.text(
            x0 + length_deg / 2.0,
            y0 + tick_h * 2.5,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            **text_kwargs,
        )

    @staticmethod
    def _station_label(ax, st, x, y, tx, ty, projection=None):
        kwargs = {"transform": projection} if projection is not None else {}
        ax.scatter(
            [x],
            [y],
            s=100 if st["is_crossing"] else 80,
            marker="v",
            c="red" if st["is_crossing"] else "gray",
            edgecolors="black",
            linewidths=2 if st["is_crossing"] else 1,
            alpha=1.0 if st["is_crossing"] else 0.6,
            zorder=8 if st["is_crossing"] else 7,
            **kwargs,
        )
        ax.text(
            tx,
            ty,
            st["name"],
            fontsize=5 if st["is_crossing"] else 4,
            ha="center",
            va="bottom",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
            ),
            alpha=1.0 if st["is_crossing"] else 0.7,
            zorder=9 if st["is_crossing"] else 8,
            clip_on=False,
            **kwargs,
        )

    @staticmethod
    def _plot_stations(ax, stations, bbox, projection=None):
        offset = (bbox["lat_max"] - bbox["lat_min"]) * 0.035
        if projection is not None:
            for st in stations:
                coord = projection.transform_point(
                    st["lon"], st["lat"], ccrs.PlateCarree()
                )
                text_coord = projection.transform_point(
                    st["lon"], st["lat"] + offset, ccrs.PlateCarree()
                )
                FlightVizPDF._station_label(
                    ax, st, coord[0], coord[1], text_coord[0], text_coord[1], projection
                )
            return

        for st in stations:
            FlightVizPDF._station_label(
                ax, st, st["lon"], st["lat"], st["lon"], st["lat"] + offset
            )

    def _draw_main_map(self, fig, flight, geom, stations):
        bbox, fz = geom["bbox"], geom["fz"]

        proj = ccrs.AlbersEqualArea(
            central_longitude=-154,
            central_latitude=50,
            standard_parallels=(55, 65),
        )
        ax = fig.add_axes([0.38, 0.48, 0.52, 0.37], projection=proj)
        ax.set_extent(
            [bbox["lon_min"], bbox["lon_max"], bbox["lat_min"], bbox["lat_max"]],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="wheat", zorder=1)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="lightblue", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            linestyle=":",
            linewidth=0.5,
            zorder=2,
        )

        coords = proj.transform_points(
            ccrs.PlateCarree(),
            flight["longitude"].values,
            flight["latitude"].values,
        )
        px, py = coords[:, 0], coords[:, 1]
        scatter = ax.scatter(
            px,
            py,
            c=fz,
            cmap="viridis",
            s=50,
            alpha=0.8,
            transform=proj,
            zorder=3,
            edgecolors="none",
        )
        ax.plot(px, py, "k-", lw=1, alpha=0.3, transform=proj, zorder=2)

        sxy = proj.transform_point(
            geom["start_lon"], geom["start_lat"], ccrs.PlateCarree()
        )
        exy = proj.transform_point(
            geom["end_lon"], geom["end_lat"], ccrs.PlateCarree()
        )
        ax.plot(
            [sxy[0], exy[0]],
            [sxy[1], exy[1]],
            "r--",
            lw=2,
            alpha=0.7,
            transform=proj,
            zorder=2,
        )

        cax = fig.add_axes([0.92, 0.48, 0.015, 0.37])
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label("Elevation (m)", fontsize=9, fontweight="bold")
        cbar.ax.tick_params(labelsize=8)

        self._marker(ax, sxy[0], sxy[1], "S", "green", transform=proj, zorder=6)
        self._marker(ax, exy[0], exy[1], "E", "red", transform=proj, zorder=6)
        self._plot_stations(ax, stations, bbox, projection=proj)

        gl = ax.gridlines(
            draw_labels=True,
            dms=False,
            x_inline=False,
            y_inline=False,
            linewidth=0.8,
            color="gray",
            alpha=0.4,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        gl.xformatter = mticker.FuncFormatter(lambda x, pos: f"{abs(x):.1f}")
        gl.yformatter = mticker.FuncFormatter(lambda y, pos: f"{abs(y):.1f}")
        self._scale_bar(ax, bbox, use_cartopy=True)

    def _draw_inset(self, fig, geom):
        proj = ccrs.AlbersEqualArea(
            central_longitude=-154,
            central_latitude=50,
            standard_parallels=(55, 65),
        )
        ax = fig.add_axes([0.18, 0.72, 0.15, 0.12], projection=proj)
        ax.set_extent([-172, -128, 51, 72], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="wheat", zorder=1)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="lightblue", zorder=0)
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            linestyle=":",
            linewidth=0.5,
            zorder=2,
        )

        self._marker(
            ax,
            geom["start_lon"],
            geom["start_lat"],
            "S",
            "green",
            transform=ccrs.PlateCarree(),
            zorder=12,
        )
        self._marker(
            ax,
            geom["end_lon"],
            geom["end_lat"],
            "E",
            "red",
            transform=ccrs.PlateCarree(),
            zorder=12,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)

    def _draw_profile(self, fig, flight, crosses, geom, speeds):
        ax = fig.add_axes([0.08, 0.12, 0.79, 0.28])
        dist_h, fz = geom["dist_h"], geom["fz"]

        scatter = ax.scatter(
            dist_h,
            fz,
            c=speeds,
            cmap="plasma",
            s=30,
            alpha=0.9,
            edgecolors="none",
            zorder=3,
        )
        ax.plot(dist_h, fz, "-", color="gray", lw=1, alpha=0.4, zorder=2)

        cax = fig.add_axes([0.92, 0.12, 0.015, 0.28])
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label("Speed (m/s)", fontsize=9, fontweight="bold")
        cbar.ax.tick_params(labelsize=8)

        for _, crossing in crosses.iterrows():
            match = self.stations[self.stations["Station"] == crossing["station"]]
            if match.empty:
                continue
            try:
                station_lon = match.iloc[0]["Longitude"]
                station_lat = match.iloc[0]["Latitude"]
                d = self.haversine_distance(
                    station_lon,
                    station_lat,
                    flight["longitude"].values,
                    flight["latitude"].values,
                )
                idx = np.argmin(d)
                x, y = dist_h[idx], fz[idx]

                ax.plot([x, x], [0, y], "r--", lw=1.5, alpha=0.6, zorder=4)
                ax.scatter(
                    [x],
                    [0],
                    s=100,
                    c="red",
                    marker="v",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=6,
                    clip_on=False,
                )
                ax.scatter(
                    [x],
                    [y],
                    s=30,
                    c="red",
                    marker="o",
                    edgecolors="black",
                    linewidths=1,
                    alpha=0.7,
                    zorder=5,
                    clip_on=False,
                )
                ax.text(
                    x,
                    fz.max() * 0.05,
                    crossing["station"],
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.5,
                    ),
                    zorder=7,
                    clip_on=False,
                    fontweight="bold",
                )
            except Exception as exc:
                print(f"Could not plot station {crossing['station']} on profile: {exc}")

        self._marker(ax, dist_h[0], fz[0], "S", "green", zorder=5)
        self._marker(ax, dist_h[-1], fz[-1], "E", "red", zorder=5)

        ax.set_xlim(dist_h.min(), dist_h.max())
        ax.set_ylim(-fz.max() * 0.1, fz.max() * 1.2)
        ax.set_xlabel("Cumulative horizontal distance (km)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Elevation (m)", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(2)

    def create_flight_page(self, pdf, flight_num, crosses, date, plot_all_stations=False):
        flight = self._read_csv(self._flight_path(date, flight_num))
        if flight is None:
            return False

        info = self.get_aircraft_info(date, flight_num)
        geom = self._geom(flight)
        speeds, max_speed = self._speed(flight, geom)
        stations = self._station_records(crosses, geom["bbox"], plot_all_stations)

        fig = plt.figure(figsize=(11, 8.5))
        self._draw_info(fig, flight_num, flight, info, geom, max_speed)
        self._draw_main_map(fig, flight, geom, stations)
        self._draw_inset(fig, geom)
        self._draw_profile(fig, flight, crosses, geom, speeds)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return True

    def generate_pdf_report(self, crossings_df, output_file=None, plot_all_stations=False):
        if crossings_df.empty:
            print("No crossings to write")
            return

        if output_file is None:
            row = crossings_df.iloc[0]
            output_file = f"{row['date']}_{row['flight_num']}_{row['equipment']}.pdf"

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(output_file) as pdf:
            pages_written = 0
            for flight_num, crosses in crossings_df.groupby("flight_num"):
                date = crosses["date"].iloc[0]
                pages_written += int(
                    self.create_flight_page(
                        pdf,
                        flight_num,
                        crosses,
                        date,
                        plot_all_stations,
                    )
                )

        print(f"Saved: {output_file}" if pages_written else "No pages written")

    def generate_grouped_pdfs(
        self,
        results,
        output_dir=None,
        plot_all_stations=False,
        suffix_func=None,
    ):
        if results.empty:
            return

        output_dir = Path(output_dir) if output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        for (date, flight_num), group in results.groupby(["date", "flight_num"]):
            equipment = group["equipment"].iloc[0]
            suffix = f"_{suffix_func(group)}" if suffix_func else ""
            filename = f"{date}_{flight_num}_{equipment}{suffix}.pdf"
            path = output_dir / filename if output_dir else Path(filename)
            try:
                self.generate_pdf_report(group, path, plot_all_stations)
            except Exception as exc:
                print(f"Failed: {flight_num} ({exc})")

    def generate_pdfs_by_date(self, date, max_crossing_km=5, output_dir=None):
        df = self._filter_crossings(
            self.crossings.copy(),
            start_date=date,
            end_date=date,
            max_crossing_km=max_crossing_km,
        )
        if df.empty:
            print("No flights found")
            return
        self.generate_grouped_pdfs(df, output_dir=output_dir)


def ask_date_range():
    start_date = input("Start date YYYYMMDD, Enter for all: ").strip() or None
    end_date = input("End date, Enter for one date: ").strip() if start_date else None
    return start_date, (end_date or start_date if start_date else None)


def ask_common_options():
    start_date, end_date = ask_date_range()
    max_dist = input("Max station distance in km, Enter to skip: ").strip()
    plot_all = input("Plot all stations y/n: ").strip().lower() == "y"
    return start_date, end_date, float(max_dist) if max_dist else None, plot_all


def maybe_generate_pdfs(viz, results, plot_all_stations=False, suffix_func=None):
    if results.empty:
        return
    viz.summarize_query_results(results)
    if input("Generate PDF y/n: ").strip().lower() != "y":
        return
    output_dir = input("Output directory, Enter for current: ").strip() or None
    viz.generate_grouped_pdfs(results, output_dir, plot_all_stations, suffix_func)


def run_interactive_tool():
    print("\nFlight Trajectory Tool\n")

    viz = FlightVizPDF(
        flight_dir=FLIGHT_DIR,
        station_file=STATION_FILE,
        crossing_file=CROSSING_FILE,
        utm_zone=6,
    )

    actions = {
        "1": "Station radius and time",
        "2": "Flight number",
        "3": "Specific flight-station crossing",
        "4": "All flights on a date",
        "5": "Exit",
    }

    while True:
        print("\nOptions")
        for key, label in actions.items():
            print(f"{key}. {label}")

        choice = input("Choice: ").strip()
        if choice == "5":
            break
        if choice not in actions:
            print("Invalid choice")
            continue

        try:
            if choice == "1":
                station_name = input("Station: ").strip().upper()
                if viz.stations[viz.stations["Station"] == station_name].empty:
                    print("Station not found")
                    continue
                radius_km = float(input("Radius km: ").strip())
                start_date, end_date, max_dist, plot_all = ask_common_options()
                results = viz.query_flights_by_station(
                    station_name=station_name,
                    radius_km=radius_km,
                    start_date=start_date,
                    end_date=end_date,
                    max_crossing_km=max_dist,
                )
                maybe_generate_pdfs(viz, results, plot_all)

            elif choice == "2":
                numbers = [x.strip() for x in input("Flight numbers: ").split(",")]
                start_date, end_date, max_dist, plot_all = ask_common_options()
                results = viz.query_flights_by_numbers(
                    numbers,
                    start_date,
                    end_date,
                    max_dist,
                )
                maybe_generate_pdfs(viz, results, plot_all)

            elif choice == "3":
                flight_num = input("Flight number: ").strip()
                station_name = input("Station: ").strip().upper()
                date = input("Date YYYYMMDD: ").strip()
                plot_all = input("Plot all stations y/n: ").strip().lower() == "y"

                result = viz.query_flight_station_crossing(
                    flight_num,
                    station_name,
                    date,
                )
                if not result.empty and input("Generate PDF y/n: ").strip().lower() == "y":
                    output_dir = input("Output directory, Enter for current: ").strip() or None
                    viz.generate_grouped_pdfs(
                        result,
                        output_dir,
                        plot_all,
                        suffix_func=lambda group: f"station_{group.iloc[0]['station']}",
                    )

            elif choice == "4":
                date = input("Date YYYYMMDD: ").strip()
                if len(date) != 8 or not date.isdigit():
                    print("Invalid date")
                    continue
                max_dist = input("Max crossing distance km, default 5: ").strip()
                output_dir = input("Output directory, Enter for current: ").strip() or None
                viz.generate_pdfs_by_date(
                    date,
                    float(max_dist) if max_dist else 5.0,
                    output_dir,
                )

        except ValueError as exc:
            print(f"Invalid input: {exc}")


if __name__ == "__main__":
    run_interactive_tool()