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
    freq_hz = waveform['samples_per_second']

    num_samples = round(et * freq_hz) - round(st * freq_hz)
    nan_values = np.full(num_samples, np.nan, dtype=np.float64)

    start_time_nano = round(st * (10 ** 9))
    period_ns = (10 ** 9) / freq_hz
    closest_i_array = np.round((read_time_data - start_time_nano) / period_ns).astype(np.int64)

    # Make sure indices are within bounds
    mask = (closest_i_array >= 0) & (closest_i_array < num_samples)
    closest_i_array = closest_i_array[mask]
    nan_values[closest_i_array] = read_value_data[mask]

    return nan_values
