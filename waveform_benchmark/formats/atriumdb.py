import numpy as np
from waveform_benchmark.formats.base import BaseFormat

from atriumdb import AtriumSDK

class AtriumDB(BaseFormat):
    """
    Example format using NPY.
    """

    def write_waveforms(self, path, waveforms):

        # Create a new local dataset using SQLite
        sdk = AtriumSDK.create_dataset(dataset_location=path)



        # Convert each channel into an array with no gaps.
        # For example: waveforms['V5'] -> {'units': 'mV', 'samples_per_second': 360, 'chunks': [{'start_time': 0.0, 'end_time': 1805.5555555555557, 'start_sample': 0, 'end_sample': 650000, 'gain': 200.0, 'samples': array([-0.065, -0.065, -0.065, ..., -0.365, -0.335,  0.   ], dtype=float32)}]}
        for name, waveform in waveforms.items():
            
            freq = waveform['samples_per_second']
            period_ns = (10 ** 9) // freq
            unit = waveform['units']
            # Define a new source.
            device_tag = "chorus"
            new_device_id = sdk.insert_device(device_tag=device_tag)
            
            for i, chunk in enumerate(waveform['chunks']):
                value_data = chunk['samples']
                # Define a new signal.
                
                new_measure_id = sdk.insert_measure(measure_tag=name, freq=freq, freq_units="Hz")

                # Write Data
                time_data = np.arange(value_data.size, dtype=np.int64) * period_ns + int(np.round(chunk['start_time'] * float(10 ** 9)))
                time_data = time_data.astype(np.int64)
            
                sdk.write_data_easy(new_measure_id, new_device_id, time_data, value_data, freq, freq_units="Hz")


    def read_waveforms(self, path, start_time, end_time, signal_names):
        
        sdk = AtriumSDK(dataset_location=path)
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
            _, read_time_data, read_value_data = sdk.get_data(measure_id=new_measure_id, start_time_n=start_time_nano, end_time_n=end_time_nano, device_id=new_device_id)

            results[signal_name] = read_value_data
        

        return results
