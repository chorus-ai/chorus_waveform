import numpy

from waveform_benchmark.formats.base import BaseFormat

class NPY(BaseFormat):
    """
    Example format using NPY for storage.
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

        for name, data in flattened_waveforms.items():
            waveform_path = f"{path}_{name}.npy"
            numpy.save(waveform_path, data)

    def read_waveforms(self, path, start_time, end_time, signal_names):
        # Read arrays from NPY files.
        results = {}
        for signal_name in signal_names:
            waveform_path = f"{path}_{signal_name}.npy"
            data = numpy.load(waveform_path, allow_pickle=True).item()

            start_sample = round(start_time / data['samples_per_second'])
            end_sample = round(end_time / data['samples_per_second'])
            samples = data['samples'][start_sample:end_sample]

            results[signal_name] = samples
        return results
