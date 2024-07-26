import numpy as np
from waveform_benchmark.formats.base import BaseFormat

from atriumdb import AtriumSDK

sdk: AtriumSDK = None
block_cache = None
filename_dict = None


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
            sdk.block.block_size = 131072
            # sdk.block.block_size = 1024
        device_tag = "chorus"
        chorus_device_id = sdk.insert_device(device_tag=device_tag)
        sdk.get_device_info(chorus_device_id)

        # Convert each channel into an array with no gaps.
        # For example: waveforms['V5'] -> {'units': 'mV', 'samples_per_second': 360, 'chunks': [{'start_time': 0.0, 'end_time': 1805.5555555555557, 'start_sample': 0, 'end_sample': 650000, 'gain': 200.0, 'samples': array([-0.065, -0.065, -0.065, ..., -0.365, -0.335,  0.   ], dtype=float32)}]}
        for name, waveform in waveforms.items():
            freq_hz = waveform['samples_per_second']
            freq_nhz = int(freq_hz * (10 ** 9))
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

            sdk.write_data_easy(measure_id, chorus_device_id, time_data, value_data, freq_nhz,
                                scale_m=scale_m, scale_b=scale_b)

    def read_waveforms(self, path, start_time, end_time, signal_names):
        assert sdk is not None, "SDK should have been initialized in writing phase"
        global block_cache
        global filename_dict
        if block_cache is None:
            block_cache, filename_dict = generate_block_cache(sdk)

        start_time_nano = int(start_time * (10 ** 9))
        end_time_nano = int(end_time * (10 ** 9))

        measures = {measure['tag']: measure['id'] for _, measure in sdk._measures.items()}
        new_device_id = sdk.get_device_id("chorus")

        # Read Data
        results = {}
        for signal_name in signal_names:
            new_measure_id = measures[signal_name]
            freq_nhz = sdk.get_measure_info(new_measure_id)['freq_nhz']

            # Get blocks from cache
            block_list = find_blocks(block_cache, new_measure_id, new_device_id, start_time_nano, end_time_nano)
            if len(block_list) == 0:
                results[signal_name] = np.array([], dtype=np.float32)
                continue

            read_list = condense_byte_read_list(block_list)
            encoded_bytes = sdk.file_api.read_file_list(read_list, filename_dict)

            # Extract the number of bytes for each block
            num_bytes_list = [row[5] for row in block_list]

            # Decode the data and separate it into headers, times, and values
            read_time_data, read_value_data, headers = sdk.block.decode_blocks(encoded_bytes, num_bytes_list, analog=True,
                                                                  time_type=1)

            results[signal_name] = (read_time_data, read_value_data)

        return results


def generate_block_cache(cache_sdk):
    cache = {}
    query = """
    SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values
    FROM block_index
    ORDER BY measure_id, device_id, start_time_n ASC;
    """

    with cache_sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute(query, ())
        block_query_result = cursor.fetchall()

    file_id_list = list(set([row[3] for row in block_query_result]))
    filename_dict = cache_sdk.get_filename_dict(file_id_list)
    for block in block_query_result:
        block_id, measure_id, device_id, file_id, start_byte, num_bytes, start_time, end_time, num_values = block
        if measure_id not in cache:
            cache[measure_id] = {}

        measure_cache = cache[measure_id]

        if device_id not in measure_cache:
            measure_cache[device_id] = []

        measure_cache[device_id].append(block)

    return cache, filename_dict


def find_blocks(cache, measure_id, device_id, start_time, end_time):
    if measure_id not in cache or device_id not in cache[measure_id]:
        return []

    blocks = cache[measure_id][device_id]

    start_idx = None
    end_idx = None

    for i, block in enumerate(blocks):
        block_start_time = block[6]
        block_end_time = block[7]

        # Find start_idx
        if start_idx is None and block_start_time > start_time:
            start_idx = i

        if start_idx is None and block_start_time <= start_time < block_end_time:
            start_idx = i

        # Find end_idx
        if block_start_time <= end_time < block_end_time:
            end_idx = i
            break
        elif block_start_time > end_time:
            end_idx = i - 1
            break

    # Handle cases where start_idx or end_idx are not set
    if start_idx is None:
        start_idx = len(blocks)
    if end_idx is None:
        end_idx = len(blocks) - 1

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
