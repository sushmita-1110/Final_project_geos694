# Final Project GEOS694

This project presents a workflow for querying aircraft crossings near seismic stations, downloading waveform data, generating spectrograms, and creating flight-path PDF reports for seismic and aircraft-analysis applications. It was developed as the final project for **GEOS 694: Introduction to Computational Geosciences (Bryant Chow)**.

## What

This project provides a structured workflow for identifying aircraft crossings near seismic stations and analyzing the associated data products. The repository includes tools for:

- reading and filtering aircraft crossing information
- downloading waveform windows from IRIS around crossing times
- generating spectrogram figures from downloaded MiniSEED files
- creating PDF reports that summarize flight geometry and station-crossing information

## How

The workflow is split into modular scripts:

- `fetch_data.py` downloads waveform data from IRIS
- `spectrogram.py` builds spectrogram products from downloaded MiniSEED files
- `flight_query.py` provides an interactive query and PDF-generation tool

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
## Acknowledgments 

Parts of this workflow were developed with guidance from and inspiration from existing code by **Bella Seppi**. Some ideas and code structure were adapted from the following repository: - uafgeotools/parkshwynodal(https://github.com/uafgeotools/parkshwynodal) 

Thank you to **Bella Seppi** for making related workflow components and research code available.

## Clone this repository

git clone git@github.com:sushmita-1110/Final_project_geos694.git

cd Final_project_geos694

## Installation

This project is intended to run in a Conda environment.

### Create environment

```bash
conda env create -f environment.yaml
conda activate aircraftseismo