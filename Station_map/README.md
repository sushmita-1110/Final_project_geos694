# Alaska (AK-network) Station Map

## Overview
This project plots Alaska AK-network seismic station locations on a topographic map using PyGMT.

The script:
- reads a station text file
- checks that the required columns are present
- plots station locations on an Alaska relief map
- labels each station
- saves the final figure as a PNG file

## Files
- `byoc_updated_station_map.py` — main Python script
- `gmap-stations-AK.txt` — input station file
- `AK_station_alaska_map.png` — output map image

## Required packages
Install the required packages before running the script:

```bash
pip install pandas
conda install -c conda-forge pygmt