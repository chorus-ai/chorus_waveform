import numpy
import wfdb


def load_wfdb_signals(record_name, pn_dir=None, start=None, end=None):
    header = wfdb.rdheader(record_name, pn_dir=pn_dir)
    if start is None:
        start = 0
    start = round(header.get_frame_number(start))
    if end is None:
        end = header.sig_len
    end = round(header.get_frame_number(end))

    record = wfdb.rdrecord(record_name, pn_dir=pn_dir, return_res=32,
                           smooth_frames=False, m2s=False,
                           sampfrom=start, sampto=end)

    if isinstance(record, wfdb.MultiRecord):
        segments = record.segments
        segment_lengths = record.seg_len
    else:
        segments = [record]
        segment_lengths = [record.sig_len]
    layout = segments[0]

    waveforms = {}

    for i, name in enumerate(layout.sig_name):
        waveform = {
            'units': layout.units[i],
            'samples_per_second': layout.fs * layout.samps_per_frame[i],
            'chunks': [],
        }
        waveforms[name] = waveform

    end_frame = 0
    for seg, seg_len in zip(segments, segment_lengths):
        start_frame = end_frame
        end_frame += seg_len
        if seg is None or seg_len == 0:
            continue
        for i, name in enumerate(seg.sig_name):
            chunk = {
                'start_time': start_frame / layout.fs,
                'end_time': end_frame / layout.fs,
                'start_sample': start_frame * layout.samps_per_frame[i],
                'end_sample': end_frame * layout.samps_per_frame[i],
                'gain': seg.adc_gain[i],
                'samples': seg.e_p_signal[i],
            }
            wave_chunks = waveforms[name]['chunks']
            if (wave_chunks
                    and wave_chunks[-1]['end_sample'] == chunk['start_sample']
                    and wave_chunks[-1]['gain'] == chunk['gain']):
                wave_chunks[-1]['end_sample'] = chunk['end_sample']
                wave_chunks[-1]['end_time'] = chunk['end_time']
                wave_chunks[-1]['samples'] = numpy.concatenate(
                    (wave_chunks[-1]['samples'], chunk['samples'])
                )
            else:
                wave_chunks.append(chunk)

    return waveforms
