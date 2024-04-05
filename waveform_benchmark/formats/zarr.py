import numpy as np
import zarr
from waveform_benchmark.formats.base import BaseFormat


class Zarr(BaseFormat):
    """
    Example format using Zarr 
    """

    def write_waveforms(self, path, waveforms):
        # Initialize Zarr group 
        root_group = zarr.open_group(path, mode='w')

        for name, waveform in waveforms.items():
            length = waveform['chunks'][-1]['end_sample']
            samples = np.empty(length, dtype=np.float32)
            samples[:] = np.nan 

            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']
                samples[start:end] = chunk['samples']

            # Create a dataset for each waveform within the root group.
            ds = root_group.create_dataset(name, data=samples, chunks=True, dtype=np.float32)
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

class Zarr_compressed(BaseFormat):
    """
    Example format using Zarr with compression.
    """

    def write_waveforms(self, path, waveforms):
        # Initialize Zarr group 
        root_group = zarr.open_group(path, mode='w')

        for name, waveform in waveforms.items():
            length = waveform['chunks'][-1]['end_sample']
            samples = np.empty(length, dtype=np.float32)
            samples[:] = np.nan 

            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']
                samples[start:end] = chunk['samples']

            # each waveform within the root group with compression.
            ds = root_group.create_dataset(name, data=samples, chunks=True, dtype=np.float32, compressor=zarr.Blosc(cname='zstd', clevel=9, shuffle=zarr.Blosc.BITSHUFFLE))
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