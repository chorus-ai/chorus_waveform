import bisect
from pathlib import Path
import json

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
            # Set the block size to 16 times less than the default.
            sdk.block.block_size = 131072 // 8
        device_tag = "chorus"
        chorus_device_id = sdk.insert_device(device_tag=device_tag)
        sdk.get_device_info(chorus_device_id)

        # Convert each channel into an array with no gaps.
        # For example: waveforms['V5'] -> {'units': 'mV', 'samples_per_second': 360, 'chunks': [{'start_time': 0.0, 'end_time': 1805.5555555555557, 'start_sample': 0, 'end_sample': 650000, 'gain': 200.0, 'samples': array([-0.065, -0.065, -0.065, ..., -0.365, -0.335,  0.   ], dtype=float32)}]}
        for name, waveform in waveforms.items():
            freq_hz = waveform['samples_per_second']
            # sdk.block.block_size = int(freq_hz * 5)  # Dynamic block size, 5 seconds per block
            freq_nhz = round(freq_hz * (10 ** 9))
            period_ns = (10 ** 18) // freq_nhz
            measure_id = sdk.insert_measure(measure_tag=name, freq=freq_hz, freq_units="Hz")

            # Convert chunks into an array with no gaps.
            sig_gain = 0
            waveform_start = None
            time_chunks, value_chunks = [], []
            for chunk in waveform['chunks']:
                value_data = chunk['samples']
                start_time_nano = int(np.round(chunk['start_time'] * float(10 ** 9)))
                waveform_start = start_time_nano if waveform_start is None else min(start_time_nano, waveform_start)

                time_data = np.arange(value_data.size, dtype=np.int64) * period_ns + start_time_nano
                time_chunks.append(time_data)
                value_chunks.append(value_data)
                sig_gain = max(sig_gain, chunk['gain'])

            if len(time_chunks) == 0:
                continue
            time_data = np.concatenate(time_chunks, dtype=np.int64)
            value_data = np.concatenate(value_chunks, dtype=value_chunks[0].dtype)

            sig_baseline = 0

            # Remove NaN values from value_data and the corresponding indices from time_data
            non_nan_indices = ~np.isnan(value_data)
            value_data = value_data[non_nan_indices]
            time_data = time_data[non_nan_indices]

            # Check if all digital values are integers
            digital_values = (value_data * sig_gain) - sig_baseline
            digital_values_are_all_ints = np.all(np.isclose(digital_values, np.round(digital_values)))

            scale_m, scale_b = None, None
            if digital_values_are_all_ints:
                value_data = np.round(digital_values).astype(np.int64)
                scale_m = 1 / sig_gain
                scale_b = float(sig_baseline) / sig_gain

            if time_data.size == 0:
                continue

            _, _, _, filename = sdk.write_data(
                measure_id, chorus_device_id, time_data, value_data, freq_nhz, int(time_data[0]),
                raw_time_type=1, raw_value_type=1, encoded_time_type=2, encoded_value_type=3,
                scale_m=scale_m, scale_b=scale_b)

            block_list, block_start_list, block_end_list = get_block_data(sdk, measure_id, chorus_device_id)
            if len(block_list) == 0:
                raise ValueError("Cannot save header information, no blocks were written")
            file_id = block_list[0][3]

            # Save header information to file
            meta_dir = Path(path) / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            header_filename = meta_dir / f"{measure_id}_{chorus_device_id}.json"
            header_data = {
                "block_list": block_list,
                "block_start_list": block_start_list,
                "block_end_list": block_end_list,
                "file_id": file_id,
                "filename": filename
            }
            with open(header_filename, 'w') as f:
                json.dump(header_data, f)

    def read_waveforms(self, path, start_time, end_time, signal_names):
        assert sdk is not None, "SDK should have been initialized in writing phase"

        start_time_nano = int(start_time * (10 ** 9))
        end_time_nano = int(end_time * (10 ** 9))

        measures = {measure['tag']: (measure['id'], measure['freq_nhz']) for _, measure in sdk._measures.items()}
        new_device_id = sdk.get_device_id("chorus")

        # Read Data
        results = {}
        for signal_name in signal_names:
            new_measure_id, freq_nhz = measures[signal_name]
            header_data = read_header_data(path, new_measure_id, new_device_id)
            if header_data is None:
                freq_hz = freq_nhz / (10 ** 9)
                start_frame = round(start_time * freq_hz)
                end_frame = round(end_time * freq_hz)
                num_samples = end_frame - start_frame
                nan_values = np.empty(num_samples, dtype=np.float32)
                if num_samples > 0:
                    nan_values[:] = np.nan
                results[signal_name] = nan_values
                continue

            filename_dict = {header_data["file_id"]: header_data["filename"]}

            # Get blocks from cache
            block_list = find_blocks(header_data["block_list"], header_data["block_start_list"],
                                     header_data["block_end_list"], start_time_nano, end_time_nano)

            if len(block_list) == 0:
                freq_hz = freq_nhz / (10 ** 9)
                start_frame = round(start_time * freq_hz)
                end_frame = round(end_time * freq_hz)
                num_samples = end_frame - start_frame
                nan_values = np.empty(num_samples, dtype=np.float32)
                if num_samples > 0:
                    nan_values[:] = np.nan
                results[signal_name] = nan_values
                continue

            read_list = condense_byte_read_list(block_list)
            encoded_bytes = sdk.file_api.read_file_list(read_list, filename_dict)

            # Extract the number of bytes for each block
            num_bytes_list = [row[5] for row in block_list]

            # Decode the data and separate it into headers, times, and values
            read_time_data, read_value_data, headers = sdk.block.decode_blocks(encoded_bytes, num_bytes_list,
                                                                               analog=True,
                                                                               time_type=1)

            # Truncate unneeded values.
            left = np.searchsorted(read_time_data, start_time_nano, side='left')
            right = np.searchsorted(read_time_data, end_time_nano, side='left')
            read_time_data, read_value_data = read_time_data[left:right], read_value_data[left:right]

            results[signal_name] = (read_time_data, read_value_data)

        return results


def get_block_data(block_sdk, measure_id, device_id):
    query = """
    SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values
    FROM block_index
    WHERE measure_id = ? AND device_id = ?
    ORDER BY measure_id, device_id, start_time_n ASC;
    """

    with block_sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute(query, (measure_id, device_id))
        block_query_result = cursor.fetchall()

    block_start_list, block_end_list = [], []
    for block in block_query_result:
        block_id, measure_id, device_id, file_id, start_byte, num_bytes, start_time, end_time, num_values = block
        block_start_list.append(start_time)
        block_end_list.append(end_time)

    return block_query_result, block_start_list, block_end_list


def read_header_data(path, measure_id, device_id):
    meta_dir = Path(path) / "meta"
    header_filename = meta_dir / f"{measure_id}_{device_id}.json"

    if not header_filename.exists():
        return None

    with open(header_filename, 'r') as f:
        header_data = json.load(f)

    return header_data


def find_blocks(blocks, starts, ends, start_time, end_time):
    start_idx = bisect.bisect_left(ends, start_time)
    end_idx = bisect.bisect_left(ends, end_time)

    if start_idx == end_idx:
        if start_idx >= len(starts):
            return []
        if (not (starts[start_idx] <= start_time <= ends[start_idx])
                and not (starts[end_idx] <= end_time <= ends[end_idx])):
            return []

    if end_idx < len(starts) and end_time < starts[end_idx]:
        end_idx = max(0, end_idx - 1)

    return blocks[start_idx:end_idx + 1]


def condense_byte_read_list(block_list):
    result = []

    for row in block_list:
        if len(result) == 0 or result[-1][2] != row[3] or result[-1][3] + result[-1][4] != row[4]:
            # append measure_id, device_id, file_id, start_byte and num_bytes
            result.append([row[1], row[2], row[3], row[4], row[5]])
        else:
            # if the blocks are continuous merge the reads together by adding the size of the next block to the
            # num_bytes field
            result[-1][4] += row[5]

    return result
