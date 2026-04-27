# Final Project GEOS694

This project presents a workflow for querying aircraft crossings near seismic stations, downloading waveform data, generating spectrograms, and creating flight-path PDF reports for seismic and aircraft-analysis applications. It was developed as the final project for **GEOS 694: Introduction to Computational Geosciences (Bryant Chow)**.

## Version: 1.0.0

## What

This project provides a structured workflow for identifying aircraft crossings near seismic stations and analyzing the associated data products. The repository includes tools for:

- reading and filtering aircraft crossing information
- downloading waveform windows from IRIS around crossing times
- generating spectrogram figures from downloaded MiniSEED files
- creating PDF reports that summarize flight geometry and station-crossing information

The repository also includes a `Station_map/` folder with a separate station-mapping exercise using PyGMT.

## How

The workflow is split into modular scripts:

- `fetch_data.py` downloads waveform data from IRIS
- `spectrogram.py` builds spectrogram products from downloaded MiniSEED files
- `flight_query.py` provides an interactive query and PDF-generation tool
- `example/example_script.py` gives a compact one-flight example that downloads waveform data, generates a spectrogram, and creates a flight-path PDF 

The code uses reusable functions and a central `FlightVizPDF` class to organize data access, querying, plotting, and PDF export.

## Why

The goal is to make the workflow easier to test, document, maintain, and expand for future aircraft-seismology analysis. Rather than writing one long script for a single figure, this project provides a reusable framework that other students can install, understand, and run.

---

## Selected project tasks

### Project Task 1

I selected the following items from Task 1:

- **Classes**  
  The `FlightVizPDF` class serves as the main structural component of the flight-query and PDF-generation workflow. It bundles related data and methods for station metadata, crossing metadata, plotting, querying, and report generation.

- **Parallel / Concurrency**  
  `fetch_data.py` uses `ThreadPoolExecutor` to download waveform windows for multiple rows more efficiently.

### Project Task 2

I selected the following items from Task 2:

- **Parameter Input System**  
  The project includes user input through the interactive menu in `flight_query.py`, where users can choose query types, dates, flight numbers, search radii, and output directories.

- **Tests / Checks**  
  The scripts use checks such as file-existence tests, empty-data checks, value filtering, and `try/except` blocks to ensure the workflow fails gracefully when data are missing or inputs are invalid.

- **State Saving**  
  The workflow saves intermediate and final products so they do not need to be recomputed each time:
  - downloaded waveforms are saved as MiniSEED files
  - spectrograms are saved as PNG files
  - flight-path reports are saved as PDF files

---

## Clone this repository

```python
git clone <github_link_to_repo>
cd Final_project_geos694
```

## Installation

This project is intended to run in a Conda environment.

### Create environment

```python
conda env create -f environment.yaml
conda activate aircraftseismo
```

## For class reviewers

The primary Python file for class review is `example/example_script.py`. The other main workflow files are `fetch_data.py`, `spectrogram.py` and `Station_map/byoc_station_map.py`, though I leave that to your preference. 

### Fetch waveform data from IRIS

```python
python fetch_data.py
```

For `fetch_data.py`, you do not need to wait for full completion, since it may take time to download a large amount of data. Running it for 1 to 2 minutes is enough for review. You may then stop it with `Ctrl + Z`. At that point, an `output/miniSEED/` folder should be created, containing waveform files organized by seismic station.

### Generate spectrograms

```python
python spectrogram.py
```

You may then run `spectrogram.py`. Again, you do not need to wait for full completion. Running it for 1 to 2 minutes is sufficient, and you may stop it with `Ctrl + Z`. After that, the `output/spectrogram/` folder should contain waveform and spectrogram PNG files organized by station.

Please feel free to explore the repository further as needed.

Note: `flight_query.py` may also be useful to review, although some of its flight-path functionality depends on flightradar24 data stored on the LUNGS scratch system, so not all parts may be directly runnable in this repository. 

## Acknowledgments 

Parts of this workflow were developed with guidance from and inspiration from existing code by **Bella Seppi**. Some ideas and code structure were adapted from the following repository: - uafgeotools/parkshwynodal(https://github.com/uafgeotools/parkshwynodal) 

Thank you to **Bella Seppi** for making related workflow components and research code available.