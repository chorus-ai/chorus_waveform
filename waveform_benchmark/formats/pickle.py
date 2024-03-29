import pickle

import numpy

from waveform_benchmark.formats.base import BaseFormat


class Pickle(BaseFormat):
    """
    Dummy example format using Pickle.
    """

    def write_waveforms(self, path, waveforms):
        # Convert each channel into an array with no gaps.
        flattened_waveforms = {}
        for name, waveform in waveforms.items():
            length = waveform['chunks'][-1]['end_sample']
            samples = numpy.empty(length, dtype=numpy.float32)
            samples[:] = numpy.nan
            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']
                samples[start:end] = chunk['samples']

            flattened_waveforms[name] = {
                'units': waveform['units'],
                'samples_per_second': waveform['samples_per_second'],
                'samples': samples,
            }

        # Dump flattened waveforms to a file.
        with open(path, 'wb') as f:
            pickle.dump(flattened_waveforms, f)

    def read_waveforms(self, path, start_time, end_time, signal_names):
        # Read arrays into memory.
        with open(path, 'rb') as f:
            channels = pickle.load(f)

        # Extract the requested samples from the array.
        results = {}
        for signal_name in signal_names:
            channel = channels[signal_name]
            start_sample = round(start_time / channel['samples_per_second'])
            end_sample = round(end_time / channel['samples_per_second'])
            results[signal_name] = channel['samples'][start_sample:end_sample]
        return results
