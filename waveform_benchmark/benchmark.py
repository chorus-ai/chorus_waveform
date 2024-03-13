#!/usr/bin/python3

import importlib
import os
import random
import tempfile
import time

from waveform_benchmark.input import load_wfdb_signals
from waveform_benchmark.ioperf import PerformanceCounter
from waveform_benchmark.utils import repeat_test
from waveform_benchmark.utils import median_attr


def run_benchmarks(input_record, format_class):
    # Load the class we will be testing
    module_name, class_name = format_class.rsplit('.', 1)
    module = importlib.import_module(module_name)
    fmt = getattr(module, class_name)

    # Load the example data
    input_record = input_record.removesuffix('.hea')
    waveforms = load_wfdb_signals(input_record)
    all_channels = list(waveforms.keys())

    total_length = 0
    timepoints_per_second = 0
    actual_samples = 0
    for waveform in waveforms.values():
        channel_length = waveform['chunks'][-1]['end_time']
        total_length = max(total_length, channel_length)
        timepoints_per_second += waveform['samples_per_second']
        actual_samples += sum(len(chunk['samples'])
                              for chunk in waveform['chunks'])
    total_timepoints = total_length * timepoints_per_second

    TEST_BLOCK_LENGTHS = [
        [total_length, 1],
        [500, 5],               # 5 random blocks of 500 seconds
        [50, 50],               # 50 random blocks of 50 seconds
        [5, 500],               # 500 random blocks of 5 seconds
    ]

    TEST_MIN_DURATION = 10
    TEST_MIN_ITERATIONS = 3

    print('_' * 64)
    print('Format: %s' % format_class)
    if fmt.__doc__:
        print('         (%s)'
              % fmt.__doc__.strip().splitlines()[0].rstrip('.'))

    print('Record: %s' % input_record)
    print('         %.0f seconds x %d channels'
          % (total_length, len(all_channels)))
    print('         %d timepoints, %d samples (%.1f%%)'
          % (total_timepoints, actual_samples,
             100 * actual_samples / total_timepoints))
    print('_' * 64)

    with tempfile.TemporaryDirectory(prefix='wavetest-', dir='.') as tempdir:
        path = os.path.join(tempdir, 'wavetest')

        # Write the example data to a file or files.
        with PerformanceCounter() as pc_write:
            fmt().write_waveforms(path, waveforms)

        # Calculate total size of the file(s).
        output_size = 0
        for subdir, dirs, files in os.walk(tempdir):
            for file in files:
                output_size += os.path.getsize(os.path.join(subdir, file))

        print('Output size:    %.0f KiB (%.2f bits/sample)'
              % (output_size / 1024, output_size * 8 / actual_samples))
        print('Time to output: %.0f sec' % pc_write.cpu_seconds)
        print('_' * 64)
        print('Read performance (median of N trials):')
        print(' #seek  #read      KiB      sec     [N]')

        for block_length, block_count in TEST_BLOCK_LENGTHS:
            counters = []
            for i in repeat_test(TEST_MIN_DURATION, TEST_MIN_ITERATIONS):
                r = random.Random(12345)
                with PerformanceCounter() as pc:
                    for j in range(block_count):
                        t0 = r.random() * (total_length - block_length)
                        t1 = t0 + block_length
                        fmt().read_waveforms(path, t0, t1, all_channels)
                counters.append(pc)

            print('%6.0f %6.0f %8.0f %8.4f  %6s read %d x %.0fs, all channels'
                  % (median_attr(counters, 'n_seek_calls'),
                     median_attr(counters, 'n_read_calls'),
                     median_attr(counters, 'n_bytes_read') / 1024,
                     median_attr(counters, 'cpu_seconds'),
                     '[%d]' % len(counters),
                     block_count,
                     block_length))

        for block_length, block_count in TEST_BLOCK_LENGTHS:
            counters = []
            r = random.Random(12345)
            for i in repeat_test(TEST_MIN_DURATION, TEST_MIN_ITERATIONS):
                with PerformanceCounter() as pc:
                    for j in range(block_count):
                        t0 = r.random() * (total_length - block_length)
                        t1 = t0 + block_length
                        c = r.choice(all_channels)
                        fmt().read_waveforms(path, t0, t1, [c])
                counters.append(pc)

            print('%6.0f %6.0f %8.0f %8.4f  %6s read %d x %.0fs, one channel'
                  % (median_attr(counters, 'n_seek_calls'),
                     median_attr(counters, 'n_read_calls'),
                     median_attr(counters, 'n_bytes_read') / 1024,
                     median_attr(counters, 'cpu_seconds'),
                     '[%d]' % len(counters),
                     block_count,
                     block_length))

    print('_' * 64)
