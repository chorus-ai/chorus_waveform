import numpy as np
from waveform_benchmark.formats.base import BaseFormat

from atriumdb import AtriumSDK


class AtriumDB(BaseFormat):
    """
    AtriumDB, a time-indexed medical waveform database.
    """

    def write_waveforms(self, path, waveforms):
        # Create a new local dataset using SQLite
        sdk = AtriumSDK.create_dataset(dataset_location=path)
        device_tag = "chorus"
        chorus_device_id = sdk.insert_device(device_tag=device_tag)

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
        sdk = AtriumSDK(dataset_location=path, num_threads=8)
        start_time_nano = int(start_time * (10 ** 9))
        end_time_nano = int(end_time * (10 ** 9))

        # get the devices
        all_devices = sdk.get_all_devices()
        all_measures = sdk.get_all_measures()
        measures = {measure['tag']: measure['id'] for _, measure in all_measures.items()}
        devices = {device['tag']: device['id'] for _, device in all_devices.items()}
        # should be a single device
        new_device_id = devices['chorus']

        # Read Data
        results = {}
        for signal_name in signal_names:
            new_measure_id = measures[signal_name]
            freq_nhz = sdk.get_measure_info(new_measure_id)['freq_nhz']
            freq_hz = freq_nhz / 10 ** 9
            period_ns = int(10 ** 18 // freq_nhz)
            start_frame = round(start_time * freq_hz)
            end_frame = round(end_time * freq_hz)
            num_samples = end_frame - start_frame

            # Since AtriumDB does not hold nan values (gaps are denoted by a jump in the time_data array)
            # We must generate a nan array to hold our result so that it can be compared to the test data.
            nan_times = np.arange(num_samples, dtype=np.int64) * period_ns + int(
                np.round(start_time * float(10 ** 9)))
            nan_values = np.empty(nan_times.size, dtype=np.float64)
            nan_values[:] = np.nan

            _, read_time_data, read_value_data = sdk.get_data(measure_id=new_measure_id, start_time_n=start_time_nano,
                                                              end_time_n=end_time_nano + period_ns,
                                                              device_id=new_device_id)

            # Write non-nan data onto nan array
            closest_i_array = np.round((read_time_data - start_time_nano) / period_ns).astype(int)

            # Make sure indices are within bounds
            mask = (closest_i_array >= 0) & (closest_i_array < num_samples)
            closest_i_array = closest_i_array[mask]
            nan_values[closest_i_array] = read_value_data[mask]

            results[signal_name] = nan_values

        return results
