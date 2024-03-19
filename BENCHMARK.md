# Benchmarking waveform formats

## Overview

The benchmarking script includes a simple set of metrics to evaluate the performance of waveform formats in Python. The benchmarks are intended to be run from the command line on Linux-systems only.

## Running a benchmark

The [waveform_benchmark.py](./waveform_benchmark.py) script is the entrypoint for running benchmarks. This script calls functions in the `waveform_benchmark` package. The syntax for running waveform_benchmark.py from the command line is: 

```
./waveform_benchmark.py <PATH_TO_RECORD> <PATH_TO_BENCHMARK_CLASS>
```

For example, to run the `WFDBFormat16` benchmark on a record named `tests/data/100`:

```
./waveform_benchmark.py ./tests/data/100 waveform_benchmark.formats.wfdb.WFDBFormat16
```

An example output is provided below:

```
________________________________________________________________
Format: waveform_benchmark.formats.wfdb.WFDBFormat16
         (WFDB with 16-bit binary storage)
Record: tests/data/100
         116147 seconds x 4 channels
         58073460 timepoints, 28425000 samples (48.9%)
________________________________________________________________
Output size:    11775 KiB (3.39 bits/sample)
Time to output: 18 sec
________________________________________________________________
Read performance (median of N trials):
 #seek  #read      KiB      sec     [N]
  1780   1563    11784   1.0967     [9] read 1 x 116147s, all channels
   300    136     1904   0.0636   [111] read 5 x 500s, all channels
  2371    908     8880   0.3640    [19] read 50 x 50s, all channels
 25298   9315    70216   3.4233     [3] read 500 x 5s, all channels
  1780   1563    11784   0.8176    [12] read 1 x 116147s, one channel
   260    115     1556   0.0532   [123] read 5 x 500s, one channel
  2478    934     9432   0.3775    [19] read 50 x 50s, one channel
 27080   9915    70004   3.5648     [3] read 500 x 5s, one channel
________________________________________________________________
```

## Adding a new format to the benchmarks

To add a new format, it is necessary to define a new Python class with two methods:

- one method to write waveform data from memory to disk
- one method to read waveform data from disk into memory

Once these methods are defined, the waveform_benchmark.py script will take care of running the benchmarks.

### Steps to add a format

#### 1. Review the following example modules:

  - [`wfdb.py`](./waveform_benchmark/formats/wfdb.py): Implements benchmarks for the WFDB format. 
  - [`pickle.py`](./waveform_benchmark/formats/pickle.py): Implements benchmarks for the Pickle format.

#### 2. Create a Python module to contain your new benchmark class.

The module should be created at `./chorus_waveform/waveform_benchmark/formats/` and named `your_format.py` (e.g. `wfdb.py`)

#### 3. Add a subclass of the [`BaseFormat`](./waveform_benchmark/formats/base.py).

The class should be named `YourFormat(BaseFormat)` (e.g. `WFDBFormat(BaseFormat)`).

#### 4. Add a `write_waveforms()` method.

This method writes a record in the target format. The method takes two arguments, `path` and `waveforms`.

  - `path` (`str`) specifies where the record will be saved. This will look something like `'./wavetest-xy2qb4_j/wavetest'`.
  - `waveforms` (`dict`) contains a waveform record, with keys representing each channel (e.g. `['MLII', 'V5']`). 
    - Each channel contains a dictionary with three keys (`'units'`, `'samples_per_second'`, `'chunks'`).
    - For example: `waveforms['V5']` -> `{'units': 'mV', 'samples_per_second': 360, 'chunks': [{'start_time': 0.0, 'end_time': 1805.5555555555557, 'start_sample': 0, 'end_sample': 650000, 'gain': 200.0, 'samples': array([-0.065, -0.065, -0.065, ..., -0.365, -0.335,  0.   ], dtype=float32)}]}`

Your method will need to transform these values into your desired format. [`pickle.py`](./waveform_benchmark/formats/pickle.py) provides a simple example, where: 

- each channel is loaded into a `numpy.array`
- the associated metadata (`units`, `samples_per_second`, `samples`) is added to a dictionary
- the record is written out to `path`.

#### 5. Add a `read_waveforms()` method.

This method reads the record written by `write_waveforms()`. The method takes four arguments, `path`, `start_time`, `end_time`, `signal_names`.

  - `path` (`str`) is the path to the waveform record saved by your `write_waveforms()` method (e.g. `'./wavetest-xy2qb4_j/wavetest'`)
  - `start_time` (`float`) is the time in seconds from where the record should be read (e.g. `0.0`).
  - `end_time` (`float`) is the time in seconds to where the record should be read (e.g. `1805.557`).
  - `signal_names` (`list`) contains the names of the channels that should be read (e.g. `['MLII', 'V5']`).

After loading the record written by `write_waveforms()`, you will need to return `results`, a `dict` with the channel names (e.g. `['MLII', 'V5']`) as keys corresponding to an array of values. For example:
  - `results.keys()` -> `dict_keys(['MLII', 'V5'])`
  - `results['MLII']` -> `array([-0.145, -0.145, -0.145, ..., -0.675, -0.765, -1.28 ], dtype=float32)`

Again, [`pickle.py`](./waveform_benchmark/formats/pickle.py) provides a simple example.

#### 6. Add your new format to the GitHub repository

Once you have created your new module, you should contribute it to the [GitHub repository](https://github.com/chorus-ai/chorus_waveform/) by opening a pull request.
