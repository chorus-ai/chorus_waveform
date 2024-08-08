import time
import numpy as np


def repeat_test(min_duration, min_iterations):
    end = time.time() + min_duration
    for _ in range(min_iterations):
        yield
    while time.time() < end:
        yield


def median_attr(objects, attr):
    values = []
    for obj in objects:
        values.append(getattr(obj, attr))
    values.sort()
    n = len(values)
    return (values[n // 2] + values[(n - 1) // 2]) / 2


def convert_time_value_pairs_to_nan_array(filedata, waveform, st, et):
    # Write non-nan data onto nan array
    read_time_data, read_value_data = filedata
    start_time_nano = int(st * (10 ** 9))
    freq_hz = waveform['samples_per_second']
    start_frame = round(st * freq_hz)
    end_frame = round(et * freq_hz)
    num_samples = end_frame - start_frame
    freq_nano = int(freq_hz * 10 ** 9)
    period_ns = (10 ** 18) // freq_nano
    nan_values = np.empty(num_samples, dtype=np.float64)
    nan_values[:] = np.nan
    closest_i_array = np.round((read_time_data - start_time_nano) / period_ns).astype(int)
    # Make sure indices are within bounds
    mask = (closest_i_array >= 0) & (closest_i_array < num_samples)
    closest_i_array = closest_i_array[mask]
    nan_values[closest_i_array] = read_value_data[mask]
    filedata = nan_values
    return filedata
