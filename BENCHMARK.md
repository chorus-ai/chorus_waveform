# Benchmarking waveform formats

## Overview

The benchmarking script includes a simple set of metrics to evaluate the performance of waveform formats in Python. The benchmarks are intended to be run from the command line on Linux-systems only.

## Running a benchmark

The [waveform_benchmark.py](./waveform_benchmark.py) script is the entrypoint for running benchmarks. This script calls functions in the `waveform_benchmark` package. The syntax for running waveform_benchmark.py from the command line is: 

```
./waveform_benchmark.py <PATH_TO_RECORD> <PATH_TO_BENCHMARK_CLASS>
```

For example, to run the `WFDBFormat516` benchmark on a record named `data/waveforms/mimic_iv/waves/p100/p10079700/85594648/85594648.hea`:

```
python ./waveform_benchmark.py ./data/waveforms/mimic_iv/waves/p100/p10079700/85594648/85594648.hea waveform_benchmark.formats.wfdb.WFDBFormat516
```

An example output is provided below:

```
________________________________________________________________
Format: waveform_benchmark.formats.wfdb.WFDBFormat516
         (WFDB with FLAC compression)
Record: ./data/waveforms/mimic_iv/waves/p100/p10079700/85594648/85594648
         214981 seconds x 6 channels
         255177600 timepoints, 199126720 samples (78.0%)
________________________________________________________________
Output size:    70744 KiB (2.91 bits/sample)
Time to output: 68 sec
________________________________________________________________
Read performance (median of N trials):
 #seek  #read      KiB      sec     [N]
  9981   9139    70768   8.2069     [3] read 1 x 214981s, all channels
   632    277     4212   0.1335    [58] read 5 x 500s, all channels
  5286   1744    19388   0.5585    [12] read 50 x 50s, all channels
 51188  16448   147048   4.6264     [3] read 500 x 5s, all channels
  6846   6359    49496   6.0602     [3] read 1 x 214981s, one channel
   240    123     1668   0.0714   [110] read 5 x 500s, one channel
  1902    738     7616   0.2974    [24] read 50 x 50s, one channel
 18932   7061    53596   2.5504     [3] read 500 x 5s, one channel
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
