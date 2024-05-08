import numpy as np
import vitaldb
from waveform_benchmark.formats.base import BaseFormat


class VitalFormat(BaseFormat):

    def write_waveforms(self, path, waveforms):
        vitalfile = vitaldb.VitalFile()
        vitalfile.dtstart = None

        for channel, waveform in waveforms.items():
            recs = []
            srate = waveform['samples_per_second']
            gain = 0
            mindisp = 9999
            maxdisp = 0

            for chunk in waveform['chunks']:
                dtstart = chunk['start_time']
                dtend = chunk['end_time']

                if vitalfile.dtstart == None or vitalfile.dtstart > dtstart:
                    vitalfile.dtstart = dtstart

                vitalfile.dtend = max(vitalfile.dtend, dtend)
                samples = chunk['samples']

                for istart in range(0, len(samples), 240000):
                    recs.append({'dt': dtstart + istart / srate,
                                 'val': samples[istart:istart+240000]})

                gain = max(gain, chunk['gain'])
                mindisp = min(mindisp, np.nanmin(samples))
                maxdisp = max(maxdisp, np.nanmax(samples))

            dtname = f"{channel}/{channel}"
            if dtname not in vitalfile.trks:
                track = vitalfile.add_track(dtname=dtname,
                                            recs=recs, srate=srate,
                                            unit=waveform['units'],
                                            mindisp=mindisp,
                                            maxdisp=maxdisp)
                track.gain = gain
            else:
                vitalfile.trks[dtname].recs.extend(recs)

        file_name = f"{path}.vital"
        vitalfile.to_vital(file_name)

    def read_waveforms(self, path, start_time, end_time, signal_names):
        signal_names = [f"{x}/{x}" for x in signal_names]
        file_name = f"{path}.vital"
        vitalfile = vitaldb.VitalFile(file_name, track_names=signal_names)
        results = {}
        for signal_name in signal_names:
            trk = vitalfile.trks[signal_name]
            if signal_name.find('/') >= 0:
                signal_name = signal_name.split('/')[-1]
            sample_length = round((end_time - start_time) * trk.srate)
            samples = np.empty(sample_length, dtype=np.float32)
            samples[:] = np.nan
            for rec in trk.recs:
                rec_start_time = rec['dt'] 
                rec_end_time = rec['dt'] + len(rec['val']) / trk.srate
                if rec_start_time > end_time:
                    break
                if start_time <= rec_start_time <= end_time or start_time <= rec_end_time <= end_time:
                    if start_time > rec_start_time and start_time <= rec_end_time < end_time:
                        crop_start = round((start_time - rec_start_time) * trk.srate)
                        rec['dt'] = start_time
                        rec_start_time = start_time
                        rec['val'] = rec['val'][crop_start:]
                    st = round((rec_start_time - start_time) * trk.srate)
                    et = min(round((rec_end_time - start_time) * trk.srate),
                            sample_length)
                    if et > st:
                        samples[st:et] = rec['val'][:et-st]
            results[signal_name] = samples
        return results
