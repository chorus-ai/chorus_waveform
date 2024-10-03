'''
Benchmark format for CCDEF, using implicit time convention (equally spaced samples according to sample rate, starting at base time).
Gain is assumed to be constant for a given channel.
'''

import h5py
import numpy as np

from waveform_benchmark.formats.base import BaseFormat


class BaseCCDEF(BaseFormat):
    """
    CCDEF signal format.
    """
    def write_waveforms(self, path, waveforms):
        """
        waveforms['V5'] -> {'units': 'mV',
                            'samples_per_second': 360,
                            'chunks': [{'start_time': 0.0,
                                        'end_time': 1805.5555555555557,
                                        'start_sample': 0,
                                        'end_sample': 650000,
                                        'gain': 200.0,
                                        'samples': array([-0.065, -0.065, -0.065, ..., -0.365, -0.335, 0. ], dtype=float32)}]
                                }
        """
        # initialize HDF5
        outputpath = path + ".hdf5"

        with h5py.File(outputpath, "w") as f:
            # create Waveform Group
            f.create_group("Waveforms")

            # loop over channels
            for channel, datadict in waveforms.items():
                # loop over chunks
                chunks = datadict["chunks"]
                # concat data
                sig_length = chunks[-1]['end_sample']
                sig_samples = np.empty(sig_length, dtype=np.short)
                nanval = -32768
                sig_samples[:] = nanval
                max_gain = max(chunk['gain'] for chunk in chunks)

                # TODO: store time as segments of sample, starttime, length

                # Determine the minimum and maximum non-NaN sample values.
                # (nanmin and nanmax will give a warning if all values are NaN.)
                rawsamples = np.concatenate([chunk['samples'] for chunk in chunks])
                sample_min = np.fmin.reduce(rawsamples)
                sample_max = np.fmax.reduce(rawsamples)
                if np.isnan(sample_min):
                    sig_baseline = 0
                else:
                    sig_baseline = round(-max_gain * (sample_min + sample_max) / 2)

                for chunk in chunks:
                    start = chunk['start_sample']
                    end = chunk['end_sample']

                    cursamples = np.where(np.isnan(chunk['samples']),
                                          (nanval*1.0)/max_gain,
                                          chunk['samples'])

                    sig_samples[start:end] = np.round(cursamples * max_gain + sig_baseline)

                if self.fmt == "Compressed":
                    f["Waveforms"].create_dataset(channel,
                                                  data=sig_samples,
                                                  compression="gzip",
                                                  compression_opts=6,
                                                  shuffle=True)
                else:
                    f["Waveforms"].create_dataset(channel,
                                                  data=sig_samples)

                f["Waveforms"][channel].attrs["uom"] = datadict["units"]
                f["Waveforms"][channel].attrs["sample_rate"] = datadict["samples_per_second"]
                f["Waveforms"][channel].attrs["nanvalue"] = nanval
                f["Waveforms"][channel].attrs["gain"] = max_gain
                f["Waveforms"][channel].attrs["baseline"] = sig_baseline
                f["Waveforms"][channel].attrs["start_time"] = chunks[0]["start_time"]

    def read_waveforms(self, path, start_time, end_time, signal_names):
        """
        Read waveforms.
        """
        outputpath = path + ".hdf5"
        results = {}

        with h5py.File(outputpath, "r") as f:
            for channel in signal_names:
                sample_rate = f["Waveforms"][channel].attrs["sample_rate"]

                # TODO: channelstarttime is unused. Remove it? 
                # channelstarttime = f["Waveforms"][channel].attrs["start_time"]

                start_frame = round((start_time) * sample_rate)
                end_frame = round((end_time) * sample_rate)

                sig_data = f["Waveforms"][channel][start_frame:end_frame] 
                naninds = (sig_data == f["Waveforms"][channel].attrs["nanvalue"])
                sig_gain = f["Waveforms"][channel].attrs["gain"]
                sig_baseline = f["Waveforms"][channel].attrs["baseline"]
                sig_data = (sig_data.astype(int) - sig_baseline) * 1.0 / sig_gain
                sig_data[naninds] = np.nan
                results[channel] = sig_data

        return results
    
    def open_waveforms(self, path: str, signal_names:list, **kwargs):
        outputpath = path + ".hdf5"
        results = {}

        try:
            f = h5py.File(outputpath, "r")
            results["file"] = f
        except FileNotFoundError:
            print(f"File not found: {outputpath}")
            results["file"] = None
        except Exception as e:
            print(f"Error processing {outputpath}: {e}")
            results["file"] = None

        return results

    def read_opened_waveforms(self, opened_files: dict, start_time: float, end_time: float,
                             signal_names: list):
        f = opened_files["file"]
        results = {}
        for channel in signal_names:
            sample_rate = f["Waveforms"][channel].attrs["sample_rate"]

            # TODO: channelstarttime is unused. Remove it? 
            # channelstarttime = f["Waveforms"][channel].attrs["start_time"]

            start_frame = round((start_time) * sample_rate)
            end_frame = round((end_time) * sample_rate)

            sig_data = f["Waveforms"][channel][start_frame:end_frame] 
            naninds = (sig_data == f["Waveforms"][channel].attrs["nanvalue"])
            sig_gain = f["Waveforms"][channel].attrs["gain"]
            sig_baseline = f["Waveforms"][channel].attrs["baseline"]
            sig_data = (sig_data.astype(int) - sig_baseline) * 1.0 / sig_gain
            sig_data[naninds] = np.nan
            results[channel] = sig_data
        return results

    def close_waveforms(self, opened_files: dict):
        opened_files["file"].close()
        opened_files.clear()


class CCDEF_Compressed(BaseCCDEF):
    """
    CCDEF compressed format.
    """
    fmt = 'Compressed'


class CCDEF_Uncompressed(BaseCCDEF):
    """
    CCDEF uncompressed format.
    """
    fmt = 'Uncompressed'
