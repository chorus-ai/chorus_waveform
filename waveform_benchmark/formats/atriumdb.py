import bisect
from collections import defaultdict
from pathlib import Path
import pickle
import itertools

import numpy as np
from waveform_benchmark.formats.base import BaseFormat

from atriumdb import AtriumSDK
from atriumdb.adb_functions import condense_byte_read_list

sdk: AtriumSDK = None

class AtriumDB(BaseFormat):
    """
    AtriumDB, a time-indexed medical waveform database.
    """
    num_threads = 1
    num_values_per_block = 16384

    def write_waveforms(self, path, waveforms):
        # Create a new local dataset using SQLite
        global sdk
        if sdk is None or path != sdk.dataset_location:
            sdk = AtriumSDK.create_dataset(dataset_location=path)
            sdk = AtriumSDK(dataset_location=path, num_threads=self.num_threads)
            sdk.block.block_size = self.num_values_per_block

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

        start_time_nano = start_time * (10 ** 9)
        end_time_nano = end_time * (10 ** 9)

        measures = {measure['tag']: (measure['id'], measure['freq_nhz']) for _, measure in sdk._measures.items()}
        new_device_id = sdk.get_device_id("chorus")

        # If the block metadata hasn't been read, read them.
        if len(sdk.block_cache) == 0:
            sdk.load_device(new_device_id)

        # Read Data
        results = {}
        for signal_name in signal_names:
            new_measure_id, freq_nhz = measures[signal_name]

            _, read_time_data, read_value_data = sdk.get_data(new_measure_id, round(start_time_nano), round(end_time_nano), device_id=new_device_id, sort=False)

            if read_value_data.size == 0:
                freq_hz = freq_nhz / (10 ** 9)
                start_frame = round(start_time * freq_hz)
                end_frame = round(end_time * freq_hz)
                num_samples = end_frame - start_frame

                results[signal_name] = np.full(num_samples, np.nan, dtype=np.float32)
                continue

            # Truncate unneeded values.
            left = np.searchsorted(read_time_data, start_time_nano, side='left')
            right = np.searchsorted(read_time_data, end_time_nano, side='left')

            if left > 0 and start_time_nano < read_time_data[left]:
                left -= 1

            read_time_data, read_value_data = read_time_data[left:right], read_value_data[left:right]

            results[signal_name] = (read_time_data, read_value_data)

        return results

    def open_waveforms(self, path: str, signal_names: list, **kwargs):
        # Strategy: Store Compressed Files In Memory
        device_id = sdk.get_device_id("chorus")
        measures = {measure['tag']: (measure['id'], measure['freq_nhz']) for _, measure in sdk._measures.items()}

        # If the block metadata hasn't been read, read them.
        if len(sdk.block_cache) == 0:
            sdk.load_device(device_id)

        # load compressed tsc files into memory:
        result = defaultdict(dict)
        for signal_name in signal_names:
            measure_id, freq_nhz = measures[signal_name]
            block_array = sdk.block_cache.get(measure_id, {}).get(device_id, [])

            # Group rows by file_id, preserving the order within each group
            for file_id, group in itertools.groupby(block_array, key=lambda row: row[3]):
                grouped_rows = list(group)

                # read the tsc file
                read_list = condense_byte_read_list(grouped_rows)
                encoded_bytes = sdk.file_api.read_file_list(read_list, sdk.filename_dict)
                result[signal_name][int(file_id)] = encoded_bytes

        return result

    def read_opened_waveforms(self, opened_files: dict, start_time: float, end_time: float,
                              signal_names: list):
        assert sdk is not None, "SDK should have been initialized in writing phase"

        start_time_nano = int(start_time * (10 ** 9))
        end_time_nano = int(end_time * (10 ** 9))

        measures = {measure['tag']: (measure['id'], measure['freq_nhz']) for _, measure in sdk._measures.items()}
        device_id = sdk.get_device_id("chorus")

        # Read Data
        results = {}
        for signal_name in signal_names:
            measure_id, freq_nhz = measures[signal_name]

            block_array = sdk.find_blocks(measure_id, device_id, start_time_nano, end_time_nano)

            if len(block_array) == 0:
                freq_hz = freq_nhz / (10 ** 9)
                start_frame = round(start_time * freq_hz)
                end_frame = round(end_time * freq_hz)
                num_samples = end_frame - start_frame

                results[signal_name] = np.full(num_samples, np.nan, dtype=np.float32)
                continue

            read_list = condense_byte_read_list(block_array)
            num_bytes_list = [row[5] for row in block_array]
            encoded_bytes = get_encoded_bytes_from_memory(opened_files[signal_name], read_list)

            read_time_data, read_value_data, _ = sdk.block.decode_blocks(
                encoded_bytes, num_bytes_list, analog=True, time_type=1, return_nan_gap=False,
                start_time_n=start_time_nano, end_time_n=end_time_nano)

            # Truncate unneeded values.
            left = np.searchsorted(read_time_data, start_time_nano, side='left')
            right = np.searchsorted(read_time_data, end_time_nano, side='left')

            if left > 0 and start_time_nano < read_time_data[left]:
                left -= 1

            read_time_data, read_value_data = read_time_data[left:right], read_value_data[left:right]

            results[signal_name] = (read_time_data, read_value_data)

        return results

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

    def read_opened_waveforms(self, opened_files: dict, start_time: float, end_time: float,
                              signal_names: list):
        assert sdk is not None, "SDK should have been initialized in writing phase"

        start_time_nano = int(start_time * (10 ** 9))
        end_time_nano = int(end_time * (10 ** 9))

        measures = {measure['tag']: (measure['id'], measure['freq_nhz']) for _, measure in sdk._measures.items()}
        device_id = sdk.get_device_id("chorus")

        # Read Data
        results = {}
        for signal_name in signal_names:
            measure_id, freq_nhz = measures[signal_name]

            block_array = sdk.find_blocks(measure_id, device_id, start_time_nano, end_time_nano)

            if len(block_array) == 0:
                freq_hz = freq_nhz / (10 ** 9)
                start_frame = round(start_time * freq_hz)
                end_frame = round(end_time * freq_hz)
                num_samples = end_frame - start_frame

                results[signal_name] = np.full(num_samples, np.nan, dtype=np.float32)
                continue

            read_list = condense_byte_read_list(block_array)
            num_bytes_list = [row[5] for row in block_array]
            encoded_bytes = get_encoded_bytes_from_memory(opened_files[signal_name], read_list)

            _, read_value_data = sdk.block.decode_blocks(
                encoded_bytes, num_bytes_list, analog=True, time_type=1, return_nan_gap=True,
                start_time_n=start_time_nano, end_time_n=end_time_nano)

            results[signal_name] = read_value_data

        return results


class AtriumDBMultiThreading(AtriumDB):
    num_threads = 40


class NanAdaptedAtriumDBMultiThreading(NanAdaptedAtriumDB):
    num_threads = 40


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


def get_encoded_bytes_from_memory(binary_data_dict, read_list):
    result = np.empty(sum([num_bytes for _, _, _, _, num_bytes in read_list]), dtype=np.uint8)

    running_index = 0
    for measure_id, device_id, file_id, start_byte, num_bytes in read_list:
        result[running_index:running_index + num_bytes] = binary_data_dict[int(file_id)][
                                                          start_byte:start_byte + num_bytes]
        running_index += num_bytes

    return result
