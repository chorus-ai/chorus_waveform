# WFDB Conversion Script

## Overview

This script converts waveform data from parquet files into WFDB (WaveForm DataBase) format. It utilizes the Sickbay Clinical Platform for signal information and saves the converted files in the specified output directory. For sites which don't use Sickbay, they'll have to modify the code to incorporate signal names, units, etc. however they store that data

## Features

- Reads waveform data from parquet files.
- Retrieves signal information from Sickbay Clinical Platform.
- Converts parquet files to WFDB format.
- Handles random patient ID and datetime shifts for anonymization.

## Requirements

- Python 3.x
- Pandas
- WFDB
- NumPy
- Sickbay
- datetime

## Installation

Install the required Python packages using pip:

```bash
pip install pandas wfdb numpy sickbay


## Configuration
base_dir: Path to the directory with raw parquet waveform files.
out_dir: Path to the directory where WFDB files will be saved.
namespace: Namespace variable to identify signal names from Sickbay.
pat_deiden_id: Random patient ID for anonymization. Adjust as needed.
rshift: Random datetime shift for the patient, measured in seconds. Adjust as needed.