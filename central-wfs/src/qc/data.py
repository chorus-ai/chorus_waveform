#!/usr/bin/python3

# taken directly from this repo:
# https://github.com/chorus-ai/chorus_waveform/blob/main/waveform_benchmark/benchmark.py

import importlib
import os
import random
import sys
import tempfile

import numpy as np

from src.utils import load_wfdb_signals
from src.qc.ioperf import PerformanceCounter
from src.utils import (
    repeat_test,
    median_attr,
    convert_time_value_pairs_to_nan_array
)

from memory_profiler import memory_usage
import time

def append_result(format_name, waveform_name, test_name, result, format_list, waveform_list, test_list, result_list):
    """
    Add a result to the summary lists
    """
    format_list.append(format_name)
    waveform_list.append(waveform_name)
    test_list.append(test_name)
    result_list.append(result)

    return format_list, waveform_list, test_list, result_list


def __read(fmt, path, total_length, all_channels, block_length, block_count, rng, test1=False):
    reader = fmt()
    for j in range(block_count):
        t0 = rng.random() * (total_length - block_length)
        t1 = t0 + block_length
        if test1:
            c = rng.choice(all_channels)
            reader.read_waveforms(path, t0, t1, [c])
        else:
            reader.read_waveforms(path, t0, t1, all_channels)

        # reads random channel and random block, opening file each time.


def _run_read_test(fmt, path, total_length, all_channels, block_length, block_count,
                   test_min_dur=10, test_min_iter=3, test1=False, mem_profile=False):
    r = random.Random(12345)
    counters = []
    mem_usages = []
    for i in repeat_test(test_min_dur, test_min_iter):
        with PerformanceCounter(mem_profile=mem_profile) as pc:
            if (mem_profile):
                mem_usage = memory_usage(
                    (__read, (fmt, path, total_length, all_channels, block_length, block_count, r), {"test1": test1}),
                    include_children=True, max_usage=True)
                mem_usages.append(mem_usage)
            else:
                __read(fmt, path, total_length, all_channels, block_length, block_count, r, test1=test1)
        counters.append(pc)
    return counters, mem_usages


# kwargs parameters are for any additional parameters that might be useful for optimization.
def __open(fmt, path, channels, **kwargs):
    reader = fmt()
    opened = reader.open_waveforms(path, channels, **kwargs)

    return (reader, opened)


def __close(reader, opened):
    reader.close_waveforms(opened)


def __read_rand_channel(reader, opened, total_length, channels, block_length, block_count, rng,
                        rand_channel: bool = False):
    # TODO  test:
    # reader = fmt()
    for j in range(block_count):
        t0 = rng.random() * (total_length - block_length)
        t1 = t0 + block_length
        if rand_channel:
            c = rng.choice(channels)
            reader.read_opened_waveforms(opened, t0, t1, [c])
        else:
            reader.read_opened_waveforms(opened, t0, t1, channels)


# reads random channel and random block, opening file(s) once.
def _run_read_test_opened_rand_channel(fmt, path, total_length, all_channels, block_length, block_count,
                                       test_min_dur=10, test_min_iter=3,
                                       rand_channel: bool = True, test1ch: bool = False, mem_profile: bool = False):
    r = random.Random(12345)
    open_counters = []
    read_counters = []
    close_counters = []
    open_mem_usages = []
    read_mem_usages = []
    close_mem_usages = []

    if (not rand_channel) and test1ch:
        channels = [r.choice(all_channels)]
    else:
        channels = all_channels
    rand_channel = test1ch and rand_channel

    for i in repeat_test(test_min_dur, test_min_iter):
        # open
        with PerformanceCounter(mem_profile=mem_profile) as pc:
            if (mem_profile):
                mem_usage, (reader, opened) = memory_usage(
                    (__open, (fmt, path, channels),
                     {'total_length': total_length, 'block_length': block_length, 'block_count': block_count}),
                    include_children=True, max_usage=True, retval=True)
                open_mem_usages.append(mem_usage)
            else:
                (reader, opened) = __open(fmt, path, channels, total_length=total_length,
                                          block_length=block_length, block_count=block_count)
        open_counters.append(pc)

        # read
        with PerformanceCounter(mem_profile=mem_profile) as pc:
            if (mem_profile):
                mem_usage = memory_usage((__read_rand_channel,
                                          (reader, opened, total_length, channels, block_length, block_count, r),
                                          {"rand_channel": rand_channel}), include_children=True, max_usage=True)
                read_mem_usages.append(mem_usage)
            else:
                __read_rand_channel(reader, opened, total_length, channels, block_length, block_count, r,
                                    rand_channel=rand_channel)
        read_counters.append(pc)

        # close
        with PerformanceCounter(mem_profile=mem_profile) as pc:
            if (mem_profile):
                mem_usage = memory_usage((__close, (reader, opened), {}), include_children=True, max_usage=True)
                close_mem_usages.append(mem_usage)
            else:
                __close(reader, opened)
        close_counters.append(pc)
    return open_counters, open_mem_usages, read_counters, read_mem_usages, close_counters, close_mem_usages


def compute_snr(reference_signal, output_signal):
    """
    Compute the signal-to-noise ratio (SNR) for the signal in decibels.
    """

    # Convert the data to NumPy arrays as needed.
    reference_signal = np.asarray(reference_signal)
    output_signal = np.asarray(output_signal)
    
    # Check that the signals have the same dimensions.
    assert (np.array_equal(np.shape(reference_signal), np.shape(output_signal)))

    # Compute the noise in the signal.
    noise_signal = output_signal - reference_signal

    # Compute the SNR with special handling for edge cases.
    x = np.sum(reference_signal ** 2)
    y = np.sum(noise_signal ** 2)

    if x > 0 and y > 0:
        snr = 10 * np.log10(x / y)
    elif y == 0:
        snr = float('inf')
    else:
        snr = float('nan')

    return snr


def print_header(mem_profile: bool = False, with_ops: bool = False):
    ops = "OP    " if with_ops else ""
    if (mem_profile):
        print('%s #seek  #read      KiB   CPU(s)    Wall(s)    Mem(MB)(used/maxrss/malloced)       [N]' % ops)
    else:
        print('%s #seek  #read      KiB   CPU(s)    Wall(s)        [N]' % ops)


def print_results(op_str: str, channels: str, counters, mem_usages, block_count, block_length,
                  mem_profile: bool = False):
    if (mem_profile):
        print('%s%6.0f %6.0f %8.0f %8.4f   %8.4f     %8.4f/%8.4f/%8.4f    %6s open %d x %.0fs, %s'
              % (op_str,
                 median_attr(counters, 'n_seek_calls'),
                 median_attr(counters, 'n_read_calls'),
                 median_attr(counters, 'n_bytes_read') / 1024,
                 median_attr(counters, 'cpu_seconds'),
                 median_attr(counters, 'walltime'),
                 # walltime / len(counters),
                 np.median(mem_usages),
                 median_attr(counters, 'max_rss'),
                 median_attr(counters, 'malloced'),
                 '[%d]' % len(counters),
                 block_count,
                 block_length,
                 channels))
    else:
        print('%s%6.0f %6.0f %8.0f %8.4f   %8.4f     %6s open %d x %.0fs, %s'
              % (op_str,
                 median_attr(counters, 'n_seek_calls'),
                 median_attr(counters, 'n_read_calls'),
                 median_attr(counters, 'n_bytes_read') / 1024,
                 median_attr(counters, 'cpu_seconds'),
                 median_attr(counters, 'walltime'),
                 '[%d]' % len(counters),
                 block_count,
                 block_length,
                 channels))

def check_data(waveforms, format_class, format_list=None, waveform_list=None, test_list=None,
                   result_list=None, mem_profile=False, test_only=False):

    # Load the class we will be testing
    module_name, class_name = format_class.rsplit('.', 1)
    module = importlib.import_module(module_name)
    fmt = getattr(module, class_name)

    with tempfile.TemporaryDirectory(prefix='wavetest-', dir='.') as tempdir:
        path = os.path.join(tempdir, 'wavetest')

        # Fidelity Check
        # Loop over each waveform
        print("Fidelity check via read_waveforms:")
        print()
        print("Chunk\t\t\tNumeric Samples\t\t\t\t  NaN Samples")
        print(f"\t# Errors  /  Total\t{'% Eq':^8}\t{'SNR':^8}\tNaN Values Match")

        for channel, waveform in waveforms.items():
            print(f"Signal: {channel}")
            # Loop over chunks
            # print("Chunk\t\t Numeric Samples\t\t  NaN Samples")
            # print(f"\t# Errors  /  Total\t{'% Eq':^8}\tNaN Values Match")

            for i_ch, chunk in enumerate(waveform["chunks"]):
                st = chunk["start_time"]
                et = chunk["end_time"]
                data = chunk["samples"]

                # read chunk from file
                filedata = fmt().read_waveforms(path, st, et, [channel])
                filedata = filedata[channel]

                # Check if the filedata is in time-value pair format
                if isinstance(filedata, tuple):
                    filedata = convert_time_value_pairs_to_nan_array(filedata, waveform, st, et)

                # compare values

                # check arrays are same size
                if data.shape != filedata.shape:
                    print(f"{i_ch:^5}\t --- Different shapes (input: {data.shape}, file: {filedata.shape}) ---")
                    continue

                # check for nans in correct location
                NANdiff = np.sum(np.isnan(data) != np.isnan(filedata))
                numnan = np.sum(np.isnan(data))
                numnanstr = f"{'N' if NANdiff else 'Y'} ({numnan})"

                # remove nans for equality check
                data_nonan = data[~np.isnan(data)]
                filedata_nonan = filedata[~np.isnan(data)]

                # use numpy's isclose to determine floating point equality
                isgood = np.isclose(filedata_nonan, data_nonan, atol=0.5 / chunk['gain'])
                numgood = np.sum(isgood)
                fpeq_rel = numgood / len(data_nonan) if len(data_nonan) != 0 else np.nan

                # compute SNR to quantify signal fidelity
                snr = compute_snr(data_nonan, filedata_nonan)

                # print to table
                print(
                    f"{i_ch:^5}\t{len(data_nonan) - numgood:10}/{len(data_nonan):10}\t{fpeq_rel * 100:^6.3f}\t\t{snr:^6.1f}\t\t{numnanstr:^16}")

                # print up to 10 bad values if not all equal
                if numgood != len(data_nonan):
                    print("Subset of unuequal numeric data from input:")
                    print(data_nonan[~isgood][:10])
                    print("Subset of unuequal numeric data from formatted file:")
                    print(filedata_nonan[~isgood][:10])
                    print(f"(Gain: {chunk['gain']})")
            # print('_' * 64)

        # Fidelity Check 2
        # Loop over each waveform

        try:
            print('_' * 64)
            print("Fidelity check via open/read/close:")
            print()
            print("Chunk\t\t\tNumeric Samples\t\t\t\t  NaN Samples")
            print(f"\t# Errors  /  Total\t{'% Eq':^8}\t{'SNR':^8}\tNaN Values Match")

            for channel,waveform in waveforms.items():
                print(f"Signal: {channel}")
                # Loop over chunks
                # print("Chunk\t\t Numeric Samples\t\t  NaN Samples")
                # print(f"\t# Errors  /  Total\t{'% Eq':^8}\tNaN Values Match")

                for i_ch, chunk in enumerate(waveform["chunks"]):
                    st = chunk["start_time"]
                    et = chunk["end_time"]
                    data = chunk["samples"]

                    # read chunk from file
                    filedata = fmt().open_read_close_waveforms(path, st, et, [channel])
                    filedata = filedata[channel]
                    
                    # compare values

                    # Check if the filedata is in time-value pair format
                    if isinstance(filedata, tuple):
                        filedata = convert_time_value_pairs_to_nan_array(filedata, waveform, st, et)

                    # check arrays are same size
                    if data.shape != filedata.shape:
                        print(f"{i_ch:^5}\t --- Different shapes (input: {data.shape}, file: {filedata.shape}) ---")
                        continue

                    # check for nans in correct location
                    NANdiff = np.sum(np.isnan(data) != np.isnan(filedata))
                    numnan = np.sum(np.isnan(data))
                    numnanstr = f"{'N' if NANdiff else 'Y'} ({numnan})"

                    # remove nans for equality check
                    data_nonan = data[~np.isnan(data)]
                    filedata_nonan = filedata[~np.isnan(data)]

                    # use numpy's isclose to determine floating point equality
                    isgood = np.isclose(filedata_nonan, data_nonan, atol=0.5/chunk['gain'])
                    numgood = np.sum(isgood)
                    fpeq_rel = numgood/len(data_nonan) if len(data_nonan) != 0 else np.nan

                    # compute SNR to quantify signal fidelity
                    snr = compute_snr(data_nonan, filedata_nonan)

                    # print to table
                    print(f"{i_ch:^5}\t{len(data_nonan)-numgood:10}/{len(data_nonan):10}\t{fpeq_rel*100:^6.3f}\t\t{snr:^6.1f}\t\t{numnanstr:^16}")

                    # print up to 10 bad values if not all equal
                    if numgood != len(data_nonan):
                        print("Subset of unuequal numeric data from input:")
                        print(data_nonan[~isgood][:10])
                        print("Subset of unuequal numeric data from formatted file:")
                        print(filedata_nonan[~isgood][:10])
                        print(f"(Gain: {chunk['gain']})")
                # print('_' * 64)
        except NotImplementedError:
            print("Skipping Fidelity check open/read/close - open once, read many - this format has not implemented the required functions.")