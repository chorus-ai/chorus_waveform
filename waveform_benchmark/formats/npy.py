import numpy

from waveform_benchmark.formats.base import BaseFormat

class NPY(BaseFormat):
    """
    Example format using NPY for storage.
    """

    def write_waveforms(self, path, waveforms):
        # Convert each channel into an array with no gaps.
        for name, waveform in waveforms.items():
            length = waveform['chunks'][-1]['end_sample']
            samples = numpy.full(length, numpy.nan, dtype=numpy.float32)

            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']
                samples[start:end] = chunk['samples']

            # Save the samples as an NPY file. 
            numpy.save(f"{path}_{name}.npy", (samples, waveform['samples_per_second']))

    def read_waveforms(self, path, start_time, end_time, signal_names):
        results = {}
        for signal_name in signal_names:
            waveform_path = f"{path}_{signal_name}.npy"
            data, fs = numpy.load(waveform_path, allow_pickle=True)

            start_sample = round(start_time * fs)
            end_sample = round(end_time * fs)
            samples = data[start_sample:end_sample]

            results[signal_name] = samples
        return results
