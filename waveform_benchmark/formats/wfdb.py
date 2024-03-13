import os

import numpy
import wfdb

from waveform_benchmark.formats.base import BaseFormat


class BaseWFDBFormat(BaseFormat):
    """
    Abstract class for WFDB signal formats.
    """

    # Currently, this class uses a single segment and stores many
    # signals in one signal file.  Using multiple segments and
    # multiple signal files could improve efficiency of storage and
    # per-channel access.

    def write_waveforms(self, path, waveforms):
        sig_name = []
        units = []
        sfreq = []
        adc_gain = []
        baseline = []
        e_p_signal = []

        length = max(waveform['chunks'][-1]['end_time']
                     for waveform in waveforms.values())

        for name, waveform in waveforms.items():
            sig_name.append(name)
            units.append(waveform['units'])
            sfreq.append(waveform['samples_per_second'])

            # Convert chunks into an array with no gaps.
            sig_length = round(length * waveform['samples_per_second'])
            sig_samples = numpy.empty(sig_length, dtype=numpy.float32)
            sig_samples[:] = numpy.nan
            sig_gain = 0
            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']
                sig_samples[start:end] = chunk['samples']
                sig_gain = max(sig_gain, chunk['gain'])

            sample_min = numpy.nanmin(sig_samples)
            sample_max = numpy.nanmax(sig_samples)
            sig_baseline = round(-sig_gain * (sample_min + sample_max) / 2)
            adc_gain.append(sig_gain)
            baseline.append(sig_baseline)
            e_p_signal.append(sig_samples)

        # Calculate frame frequency as GCD of sampling frequencies.
        ffreqs = sorted(set(sfreq))
        while len(ffreqs) > 1 and ffreqs[-1] < ffreqs[-2] * 10000:
            ffreqs[-1] %= ffreqs[-2]
            ffreqs.sort()
        fs = ffreqs[-1]
        samps_per_frame = [round(f / fs) for f in sfreq]

        # Write out data as a WFDB record.
        rec = wfdb.Record(
            record_name=os.path.basename(path),
            n_sig=len(sig_name),
            fmt=[self.fmt] * len(sig_name),
            fs=fs,
            samps_per_frame=samps_per_frame,
            sig_name=sig_name,
            units=units,
            adc_gain=adc_gain,
            baseline=baseline,
            e_p_signal=e_p_signal,
        )
        rec.e_d_signal = rec.adc(expanded=True)
        rec.set_d_features(expanded=True)
        rec.set_defaults()
        rec.wrsamp(write_dir=os.path.dirname(path), expanded=True)

    def read_waveforms(self, path, start_time, end_time, signal_names):
        header = wfdb.rdheader(path)
        start_frame = round(start_time * header.fs)
        end_frame = round(end_time * header.fs)

        record = wfdb.rdrecord(path, sampfrom=start_frame, sampto=end_frame,
                               channel_names=signal_names, smooth_frames=False,
                               return_res=32)
        results = {}
        for signal_name, samples in zip(record.sig_name, record.e_p_signal):
            results[signal_name] = samples
        return results


class WFDBFormat16(BaseWFDBFormat):
    """
    WFDB with 16-bit binary storage.
    """
    fmt = '16'


class WFDBFormat516(BaseWFDBFormat):
    """
    WFDB with FLAC compression.
    """
    fmt = '516'
