import numpy
from waveform_benchmark.formats.base import BaseFormat

class BaseNPY(BaseFormat):
    """
    Example format using NPY.
    """

    def write_waveforms(self, path, waveforms):
        # Convert each channel into an array with no gaps.
        for name, waveform in waveforms.items():
            length = waveform['chunks'][-1]['end_sample']
            samples = numpy.full(length + 1, numpy.nan, dtype=numpy.float32)

            # Store the sampling rate as the first element
            samples[0] = waveform['samples_per_second']

            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']

                # Shift the indices by 1 to accommodate the sampling rate
                samples[start + 1:end + 1] = chunk['samples']

            # Save the samples as an NPY file.
            if self.fmt == 'Compressed':
                numpy.savez_compressed(f"{path}_{name}.npz", samples)
            else:
                numpy.save(f"{path}_{name}.npy", samples)

    def read_waveforms(self, path, start_time, end_time, signal_names):
        results = {}
        for signal_name in signal_names:
            if self.fmt == 'Compressed':
                waveform_path = f"{path}_{signal_name}.npz"
                samples = numpy.load(waveform_path, mmap_mode='r')['arr_0']
            else:
                waveform_path = f"{path}_{signal_name}.npy"
                samples = numpy.load(waveform_path, mmap_mode='r')

            # Extract the sampling rate from the first element
            fs = samples[0]
            
            # exclude the sampling rate from the samples
            samples = samples[1:]

            start_sample = round(start_time * fs)
            end_sample = round(end_time * fs)
            samples = samples[start_sample:end_sample]

            results[signal_name] = samples

        return results
    

    def open_waveforms(self, path: str, signal_names:list, **kwargs):
        output = {}
        for signal_name in signal_names:
            if self.fmt == 'Compressed':
                waveform_path = f"{path}_{signal_name}.npz"
                output[signal_name] = numpy.load(waveform_path, mmap_mode='r')['arr_0']
            else:
                waveform_path = f"{path}_{signal_name}.npy"
                output[signal_name] = numpy.load(waveform_path, mmap_mode='r')
        return output
    
    def read_opened_waveforms(self, opened_files: dict, start_time: float, end_time: float,
                             signal_names: list):
        results = {}
        for signal_name in signal_names:
            samples = opened_files[signal_name]

            # Extract the sampling rate from the first element
            fs = samples[0]
            
            # exclude the sampling rate from the samples
            samples = samples[1:]

            start_sample = round(start_time * fs)
            end_sample = round(end_time * fs)
            samples = samples[start_sample:end_sample]

            results[signal_name] = samples

        return results

    def close_waveforms(self, opened_files: dict):
        opened_files.clear()

class NPY_Compressed(BaseNPY):
    fmt = 'Compressed'

class NPY_Uncompressed(BaseNPY):
    fmt = 'Uncompressed'
