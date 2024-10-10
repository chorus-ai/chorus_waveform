import numpy as np
import zarr

from waveform_benchmark.formats.base import BaseFormat


class Zarr(BaseFormat):
    """
    Example format using Zarr with 16-bit integer waveforms.
    """
    def write_waveforms(self, path, waveforms):
        # Initialize Zarr group
        root_group = zarr.open_group(path, mode='w')

        for name, waveform in waveforms.items():
            length = waveform['chunks'][-1]['end_sample']
            samples = np.empty(length, dtype=np.int16)
            samples[:] = 0

            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']
                samples[start:end] = chunk['samples']

            # Create a dataset for each waveform within the root group.
            ds = root_group.create_dataset(name, data=samples, chunks=True, dtype=np.int16)  
            ds.attrs['units'] = waveform['units']
            ds.attrs['samples_per_second'] = waveform['samples_per_second']

    def read_waveforms(self, path, start_time, end_time, signal_names):
        # Open the Zarr group
        root_group = zarr.open_group(path, mode='r')

        results = {}
        for signal_name in signal_names:
            ds = root_group[signal_name]
            samples_per_second = ds.attrs['samples_per_second']

            start_sample = round(start_time * samples_per_second)
            end_sample = round(end_time * samples_per_second)

            # Random access the Zarr array
            results[signal_name] = ds[start_sample:end_sample]

        return results
    
    def open_waveforms(self, path: str, signal_names: list, **kwargs):
        """
        Open Zarr waveforms.
        """
        output = {}
        root_group = zarr.open_group(path, mode='r')
        for signal_name in signal_names:
            output[signal_name] = root_group[signal_name]
        return output

    def read_opened_waveforms(self, opened_files: dict, start_time: float, end_time: float,
                              signal_names: list):
        """
        Read the already opened Zarr waveforms between `start_time` and `end_time`.
        """
        results = {}
        for signal_name in signal_names:
            ds = opened_files[signal_name]

            # Extract the sampling rate from the attributes of the Zarr dataset
            fs = ds.attrs['samples_per_second']
            
            start_sample = round(start_time * fs)
            end_sample = round(end_time * fs)
            
            # Random access the Zarr array
            samples = ds[start_sample:end_sample]

            results[signal_name] = samples

        return results

    def close_waveforms(self, opened_files: dict):
        """
        Clear references to the opened Zarr files.
        """
        opened_files.clear()

class Zarr_compressed(BaseFormat):
    """
    Example format using Zarr with compression and 16-bit integer waveforms.
    """

    def write_waveforms(self, path, waveforms):
        # Initialize Zarr group 
        root_group = zarr.open_group(path, mode='w')

        for name, waveform in waveforms.items():
            length = waveform['chunks'][-1]['end_sample']
            samples = np.empty(length, dtype=np.int16)
            samples[:] = np.nan 

            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']
                samples[start:end] = chunk['samples']

            # each waveform within the root group with compression.
            ds = root_group.create_dataset(name, data=samples, chunks=True, dtype=np.int16, 
                                           compressor=zarr.Blosc(cname='zstd', clevel=9, shuffle=zarr.Blosc.BITSHUFFLE)) 
            ds.attrs['units'] = waveform['units']
            ds.attrs['samples_per_second'] = waveform['samples_per_second']


    def read_waveforms(self, path, start_time, end_time, signal_names):
        # Open the Zarr group
        root_group = zarr.open_group(path, mode='r')

        results = {}
        for signal_name in signal_names:
            ds = root_group[signal_name]
            samples_per_second = ds.attrs['samples_per_second']
            
            start_sample = round(start_time * samples_per_second)
            end_sample = round(end_time * samples_per_second)

            # Random access the Zarr array 
            results[signal_name] = ds[start_sample:end_sample]

        return results