import numpy as np
import zarr
from waveform_benchmark.formats.base import BaseFormat

class BaseZarr(BaseFormat):
    """
    Example format using Zarr with 16-bit integer waveforms.
    """
    def write_waveforms(self, path, waveforms):
        # Initialize Zarr group
        root_group = zarr.open_group(path, mode='w')
        nanval = -32768  # Sentinel value for NaN

        for name, waveform in waveforms.items():
            length = waveform['chunks'][-1]['end_sample']
            samples = np.empty(length, dtype=np.int16)
            samples[:] = nanval  

            max_gain = max(chunk['gain'] for chunk in waveform['chunks'])  # Get max gain from the chunks

            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']
                # Replace NaN values in the chunk with sentinel value
                cursamples = np.where(np.isnan(chunk['samples']), nanval, np.round(chunk['samples'] * chunk['gain'])).astype(np.int16)
                samples[start:end] = cursamples

            if self.fmt == 'Compressed':
                ds = root_group.create_dataset(
                    name, 
                    data=samples, 
                    chunks=True, 
                    dtype=np.int16, 
                    compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
                )
            else:
                ds = root_group.create_dataset(name, data=samples, chunks=True, dtype=np.int16)

            ds.attrs['units'] = waveform['units']
            ds.attrs['samples_per_second'] = waveform['samples_per_second']
            ds.attrs['nanvalue'] = nanval  # Store the sentinel value for NaN
            ds.attrs['gain'] = max_gain  # Store the gain

    def read_waveforms(self, path, start_time, end_time, signal_names):
        # Open the Zarr group
        root_group = zarr.open_group(path, mode='r')

        results = {}
        for signal_name in signal_names:
            ds = root_group[signal_name]
            samples_per_second = ds.attrs['samples_per_second']
            nanval = ds.attrs['nanvalue']  # Retrieve the sentinel value for NaN
            gain = ds.attrs['gain']  # Retrieve the gain

            start_sample = round(start_time * samples_per_second)
            end_sample = round(end_time * samples_per_second)

            # Random access the Zarr array
            sig_data = ds[start_sample:end_sample]
            naninds = (sig_data == nanval)
            sig_data = sig_data.astype(np.float32)
            sig_data = sig_data / gain
            sig_data[naninds] = np.nan

            results[signal_name] = sig_data

        return results

    def open_waveforms(self, path: str, signal_names: list, **kwargs):
        """
        Open Zarr waveforms for multiple signal names.
        """
        output = {}
        root_group = zarr.open_group(path, mode='r')
        for signal_name in signal_names:
            output[signal_name] = root_group[signal_name]
        return output

    def read_opened_waveforms(self, opened_files: dict, start_time: float, end_time: float, signal_names: list):
        """
        Read the already opened Zarr.
        """
        results = {}

        for signal_name in signal_names:
            ds = opened_files[signal_name]

            # Extract the sampling rate and other attributes
            samples_per_second = ds.attrs['samples_per_second']
            nanval = ds.attrs['nanvalue']  # Retrieve the sentinel value for NaN
            gain = ds.attrs['gain']  # Retrieve the gain

            start_sample = round(start_time * samples_per_second)
            end_sample = round(end_time * samples_per_second)

            sig_data = ds[start_sample:end_sample]
            naninds = (sig_data == nanval)
            sig_data = sig_data.astype(np.float32)
            sig_data = sig_data / gain
            sig_data[naninds] = np.nan
            results[signal_name] = sig_data

        return results

    def close_waveforms(self, opened_files: dict):
        """
        Clear opened Zarr files.
        """
        opened_files.clear()

class Zarr_Compressed(BaseZarr):
    fmt = 'Compressed'

class Zarr_Uncompressed(BaseZarr):
    fmt = 'Uncompressed'
    