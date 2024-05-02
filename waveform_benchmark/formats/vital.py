import os
import numpy as np
import vitaldb
from waveform_benchmark.formats.base import BaseFormat

class VitalFormat(BaseFormat):
    def write_waveforms(self, path, waveforms):
        vitalfile = vitaldb.VitalFile()
        for name, waveform in waveforms.items():
            recs = []
            for chunk in waveform['chunks']:
                if vitalfile.dtstart == 0 or vitalfile.dtstart > chunk['start_time']:
                    vitalfile.dtstart = chunk['start_time']
                if vitalfile.dtend < chunk['end_time']:
                    vitalfile.dtend = chunk['end_time']
                rec = {'dt': chunk['start_time']}
                rec['val'] = chunk['samples']
                recs.append(rec)
            track = vitalfile.add_track(name, recs, waveform['samples_per_second'], waveform['units'])
            track.gain = waveform['gain']
        vitalfile.to_vital(path)
    def read_waveforms(self, path, start_time, end_time, signal_names):
        vitalfile = vitaldb.VitalFile(path, track_names=signal_names)
        vitalfile.crop(start_time, end_time)
        results = {}
        for trk in vitalfile.trks:
            channel = {}
            channel['units'] = trk.unit
            channel['samples_per_second'] = trk.srate
            channel['chunks'] = []
            for rec in trk.recs:
                chunk = {}
                chunk['start_time'] = rec['dt']
                chunk['end_time'] = rec['dt'] + (rec['val'].size / trk.srate)
                chunk['samples'] = rec['val']
                channel['chunks'].append(chunk)
            results[trk.dtname] = channel
        return results