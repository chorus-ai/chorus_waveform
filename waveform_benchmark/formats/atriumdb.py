import bisect
from pathlib import Path
import pickle

import numpy as np
from waveform_benchmark.formats.base import BaseFormat

from atriumdb import AtriumSDK

sdk: AtriumSDK = None

class AtriumDB(BaseFormat):
    """
    AtriumDB, a time-indexed medical waveform database.
    """

    def write_waveforms(self, path, waveforms):
        # Create a new local dataset using SQLite
        global sdk
        if sdk is None:
            sdk = AtriumSDK.create_dataset(dataset_location=path)
            sdk = AtriumSDK(dataset_location=path, num_threads=1)
            sdk.block.block_size = 16384

        device_tag = "chorus"
        chorus_device_id = sdk.insert_device(device_tag=device_tag)
        sdk.get_device_info(chorus_device_id)

        # Convert each channel into an array with no gaps.
        for name, waveform in waveforms.items():
            if len(waveform['chunks']) == 0:
                continue
            with sdk.write_buffer() as buf:
                freq_hz = waveform['samples_per_second']
                measure_id = sdk.insert_measure(measure_tag=name, freq=freq_hz, freq_units="Hz")

                # Calculate digital to analog scale factors.
                sig_gain = max(chunk['gain'] for chunk in waveform['chunks'])
                sig_baseline = 0

                scale_m = 1 / sig_gain
                scale_b = float(sig_baseline) / sig_gain

                for chunk in waveform['chunks']:
                    value_data = chunk['samples']
                    start_time_s = chunk['start_time']
                    for segment_start_s, segment_values in generate_non_nan_slices(start_time_s, freq_hz, value_data):
                        # convert analog data to digital for optimal storage
                        digital_values = (segment_values * sig_gain) - sig_baseline
                        digital_values = np.round(digital_values).astype(np.int64)
                        sdk.write_segment(measure_id, chorus_device_id, digital_values, segment_start_s, freq=freq_hz,
                                          freq_units="Hz", time_units="s", scale_m=scale_m, scale_b=scale_b)

    def read_waveforms(self, path, start_time, end_time, signal_names):
        assert sdk is not None, "SDK should have been initialized in writing phase"

        start_time_nano = int(start_time * (10 ** 9))
        end_time_nano = int(end_time * (10 ** 9))

        measures = {measure['tag']: (measure['id'], measure['freq_nhz']) for _, measure in sdk._measures.items()}
        new_device_id = sdk.get_device_id("chorus")

        # If the block metadata hasn't been read, read them.
        if len(sdk.block_cache) == 0:
            sdk.load_device(new_device_id)

        # Read Data
        results = {}
        for signal_name in signal_names:
            new_measure_id, freq_nhz = measures[signal_name]

            _, read_time_data, read_value_data = sdk.get_data(new_measure_id, start_time_nano, end_time_nano, device_id=new_device_id, sort=False)

            if read_value_data.size == 0:
                freq_hz = freq_nhz / (10 ** 9)
                start_frame = round(start_time * freq_hz)
                end_frame = round(end_time * freq_hz)
                num_samples = end_frame - start_frame
                nan_values = np.empty(num_samples, dtype=np.float32)
                if num_samples > 0:
                    nan_values[:] = np.nan
                results[signal_name] = nan_values
                continue

            # Truncate unneeded values.
            left = np.searchsorted(read_time_data, start_time_nano, side='left')
            right = np.searchsorted(read_time_data, end_time_nano, side='left')
            read_time_data, read_value_data = read_time_data[left:right], read_value_data[left:right]

            results[signal_name] = (read_time_data, read_value_data)

        return results

    def open_waveforms(self, path: str, signal_names:list, **kwargs):
        return {signal_name: np.array([]) for signal_name in signal_names}

    def read_opened_waveforms(self, opened_files: dict, start_time: float, end_time: float,
                             signal_names: list):
        return {signal_name: np.array([]) for signal_name in signal_names}

    def close_waveforms(self, opened_files: dict):
        return


class NanAdaptedAtriumDB(AtriumDB):
    """
    AtriumDBWithNan: Subclass of AtriumDB to convert time, value tuples into a single NaN array.
    """

    def read_waveforms(self, path, start_time, end_time, signal_names):
        assert sdk is not None, "SDK should have been initialized in writing phase"

        start_time_nano = int(start_time * (10 ** 9))
        end_time_nano = int(end_time * (10 ** 9))

        measures = {measure['tag']: (measure['id'], measure['freq_nhz']) for _, measure in sdk._measures.items()}
        new_device_id = sdk.get_device_id("chorus")

        # If the block metadata hasn't been read, read them.
        if len(sdk.block_cache) == 0:
            sdk.load_device(new_device_id)

        # Read Data
        results = {}
        for signal_name in signal_names:
            new_measure_id, freq_nhz = measures[signal_name]

            _, read_value_data = sdk.get_data(
                new_measure_id, start_time_nano, end_time_nano, device_id=new_device_id, return_nan_filled=True)

            results[signal_name] = read_value_data

        return results

def generate_non_nan_slices(start_time_s: float, freq_hz: float, data: np.ndarray):
    indices = np.arange(data.size, dtype=np.int32)
    values_mask = ~np.isnan(data)

    data = data[values_mask]
    indices = indices[values_mask]

    if indices.size == 0:
        return

    slices = np.concatenate([np.array([0]), np.where(np.diff(indices) > 1)[0] + 1, np.array([indices.shape[0]])])

    for i in range(slices.size - 1):
        left_index = slices[i]
        right_index = slices[i+1]
        start_time_slice = start_time_s + (indices[left_index] / freq_hz)
        yield start_time_slice, data[left_index:right_index]
