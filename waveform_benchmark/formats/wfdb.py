import os

import numpy
import soundfile
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

            # Determine the minimum and maximum non-NaN sample values.
            # (nanmin and nanmax will give a warning if all values are NaN.)
            sample_min = numpy.fmin.reduce(sig_samples)
            sample_max = numpy.fmax.reduce(sig_samples)
            if numpy.isnan(sample_min):
                sig_baseline = 0
            else:
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

    def open_waveforms(self, path, signal_names, **kwargs):
        header = wfdb.rdheader(path)
        dir_name = os.path.dirname(path)
        readers = {}
        sig_readers = {}

        # Open a reader for each signal file of interest.  For each
        # individual signal, determine the corresponding reader object
        # and channel index within that signal file.
        for name in signal_names:
            i = header.sig_name.index(name)
            file_name = header.file_name[i]
            try:
                reader = readers[file_name]
            except KeyError:
                reader = self._open_reader(header, dir_name, file_name)
                readers[file_name] = reader
            channel = reader.channels.index(name)
            sig_readers[name] = (reader, channel)

        return {
            # Frame frequency
            'fs': header.fs,

            # List of all readers
            'readers': list(readers.values()),

            # Dictionary mapping signal names to (reader, channel)
            'sig_readers': sig_readers,
        }

    def _open_reader(self, header, dir_name, file_name):
        path = os.path.join(dir_name, file_name)

        rec_channels = [
            i for i, name in enumerate(header.file_name) if name == file_name
        ]

        for i in rec_channels:
            assert header.fmt[i] == self.fmt, "incorrect format"

        channels = [header.sig_name[i] for i in rec_channels]
        spf = [header.samps_per_frame[i] for i in rec_channels]
        inv_gain = numpy.array(
            [1 / header.adc_gain[i] for i in rec_channels],
            dtype=numpy.float32,
        )
        baseline = numpy.array(
            [header.baseline[i] for i in rec_channels],
            dtype=numpy.float32,
        )

        return self.Reader(path, channels, spf, inv_gain, baseline)

    def close_waveforms(self, opened_files):
        # Close all of the readers we opened above
        for reader in opened_files['readers']:
            reader.close()

    def read_opened_waveforms(self, opened_files, start_time, end_time,
                              signal_names):
        # Determine start/end frame number
        fs = opened_files['fs']
        start_frame = round(start_time * fs)
        end_frame = round(end_time * fs)

        # Read all samples (from selected signal files) for that range
        # of frame numbers
        for reader in opened_files['readers']:
            reader.load_frames(start_frame, end_frame)

        # Extract the desired signals and return a dictionary of
        # arrays
        results = {}
        for name in signal_names:
            reader, channel = opened_files['sig_readers'][name]
            results[name] = reader.get_channel(channel)
        return results


class WFDBFormat16(BaseWFDBFormat):
    """
    WFDB with 16-bit binary storage.
    """
    fmt = '16'

    class Reader:
        def __init__(self, path, channels, spf, inv_gain, baseline):
            self.fp = open(path, 'rb')
            self.inv_gain = inv_gain
            self.baseline = baseline
            self.channels = channels

            channel_start = numpy.cumsum([0] + spf)
            self.channel_slice = [
                slice(x, y) for x, y in zip(channel_start, channel_start[1:])
            ]
            self.total_spf = channel_start[-1]
            self.frame_dtype = numpy.dtype('<i2') * self.total_spf
            self.bytes_per_frame = self.frame_dtype.itemsize

        def close(self):
            self.fp.close()

        def load_frames(self, start_frame, end_frame):
            self.fp.seek(start_frame * self.bytes_per_frame)
            self.data = numpy.fromfile(self.fp, self.frame_dtype,
                                       end_frame - start_frame)

        def get_channel(self, channel):
            data = self.data[:, self.channel_slice[channel]]
            result = data - self.baseline[channel]
            result *= self.inv_gain[channel]
            result[data == -32768] = numpy.nan
            return result.reshape(-1)


class WFDBFormat516(BaseWFDBFormat):
    """
    WFDB with FLAC compression.
    """
    fmt = '516'

    class Reader:
        def __init__(self, path, channels, spf, inv_gain, baseline):
            self.fp = soundfile.SoundFile(path)
            self.spf = spf[0]
            self.inv_gain = inv_gain
            self.baseline = baseline
            self.channels = channels

        def close(self):
            self.fp.close()

        def load_frames(self, start_frame, end_frame):
            self.fp.seek(start_frame * self.spf)
            # Note that the following call may fail, for very large
            # numbers of samples, if you are using an outdated version
            # of libsndfile (prior to 1.2.0.)
            self.data = self.fp.read((end_frame - start_frame) * self.spf,
                                     dtype='int16', always_2d=True)

        def get_channel(self, channel):
            data = self.data[:, channel]
            result = data - self.baseline[channel]
            result *= self.inv_gain[channel]
            result[data == -32768] = numpy.nan
            return result
