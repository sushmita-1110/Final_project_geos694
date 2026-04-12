"""
Plot Alaska (AK) seismic stations on a topographic map using PyGMT.

This script reads station coordinates from text file
and generates a map of Alaska showing station locations and labels.

Input
-----
- gmap-stations-AK.txt
  Text file containing at least the following columns:
    - Station
    - Latitude
    - Longitude

Output
------
- AK_station_alaska_map.png
  A PNG image of the Alaska AK(network) station map.

How to run
----------
1. Install the required packages:
   pip install pandas
   conda install -c conda-forge pygmt

2. Place the input file `gmap-stations-AK.txt` in the same directory
   as this script.

3. Run the script:
   
"""

from pathlib import Path

import pandas as pd
import pygmt


# Wrapped settings into a class
class AlaskaStationMap:
    """
    Create a topographic map of Alaska seismic stations.
    """

    def __init__(self, input_file: str, output_file: str):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.region = [-170, -135, 51, 70]
        self.projection = "L-150/62/55/65/12c"
        self.relief_grid = "@earth_relief_03m"
        self.map_title = "Alaska Stations"

    def load_station_data(self) -> pd.DataFrame:
        """
        Load station data from a pipe-delimited text file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing station names, latitudes, and longitudes.
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        station_df = pd.read_csv(self.input_file, sep="|")

        required_columns = {"Station", "Latitude", "Longitude"}
        missing_columns = required_columns - set(station_df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(missing_columns))}"
            )

        return station_df

    def create_station_map(self, station_df: pd.DataFrame) -> pygmt.Figure:
        """
        Create a PyGMT figure showing Alaska station locations.
        """
        figure = pygmt.Figure()

        figure.grdimage(
            grid=self.relief_grid,
            region=self.region,
            projection=self.projection,
            shading=True,
            cmap="geo",
        )

        figure.coast(
            region=self.region,
            projection=self.projection,
            shorelines="0.6p,black",
            borders=["1/0.5p,black", "2/0.25p,gray40"],
            rivers="a/0.25p,blue",
            lakes="lightblue",
            resolution="i",
        )

        figure.plot(
            x=station_df["Longitude"],
            y=station_df["Latitude"],
            style="i0.14c",
            fill="red",
            pen="0.25p,black",
        )

        figure.text(
            x=station_df["Longitude"],
            y=station_df["Latitude"],
            text=station_df["Station"],
            font="1.5p,Helvetica-Bold,black",
            justify="LT",
            offset="0.05c/0.05c",
            fill="white",
        )

        figure.basemap(frame=["af", f"+t{self.map_title}"])

        return figure

    def save_and_show_map(self, figure: pygmt.Figure) -> None:
        """
        Save the map to disk and display it.
        """
        figure.savefig(self.output_file, dpi=600)
        figure.show()

    # Added one workflow method
    def run(self) -> None:
        station_df = self.load_station_data()
        figure = self.create_station_map(station_df)
        self.save_and_show_map(figure)


def main():
    # Parameter input system using input()
    default_input_file = "gmap-stations-AK.txt"
    default_output_file = "AK_station_alaska_map.png"

    input_file = input(
        f"Enter input station file [{default_input_file}]: "
    ).strip()
    if input_file == "":
        input_file = default_input_file

    output_file = input(
        f"Enter output image file [{default_output_file}]: "
    ).strip()
    if output_file == "":
        output_file = default_output_file

    station_map = AlaskaStationMap(
        input_file=input_file,
        output_file=output_file,
    )
    station_map.run()


if __name__ == "__main__":
    main()