# Submissions are assumed to be formatted as pair-based OR stream-based:
# Pair-based: one timestamp for each sample
# Stream-based: more than one sample per row
    # with a corresponding "start" time and "end" time for the stream.
    # Indicates whether the start and end times correspond
    # to the beginning or end of the first and last samples.

from typing import List, Array

def detect_gaps_pair(timestamps: List, waveforms: Array):
    """
    Gap detection for pair-based waveform data.
    
    :param timestamps: (N,1) shape, for N rows/messages
        and 1 timestamp for each message/sample.
    :type timestamps: Array
    :param waveforms: Data values and types of waveform data.
    :type waveforms: Array


    Returns:
        summary_gaps: dict describing gap type(s), frequency, location(s).
        identified_gaps: list of gap indices and segments to correct for.
    """
    pass

def detect_gaps_stream(timestamps: Array, waveforms: Array ):
    """
    Gap detection for stream-based waveform data.
    
    :param timestamps: (N,2) shape, for N rows/messages
        and 2 timestamps for start and end.
        If initialy data has more than 2 timestamp column,
        then declare 2 as the bounds of each message.
    :type timestamps: Array
    :param waveforms: Data values and types of waveform data.
    :type waveforms: Array

    Returns:
        summary_gaps: dict describing gap type(s), frequency, location(s)
        identified_gaps: list of gap indices and segments to correct for
    """
    pass

def correct_gaps_pair():
    """
    Correct gaps in pair-based waveform data.

    Returns:
        corrected_timestamps: array with gaps corrected
        corrected_waveforms: waveform array with interpolated or filled values
    """
    pass

def correct_gaps_stream():
    """
    Correct gaps in stream-based waveform data.

    Returns:
        corrected_timestamps: array with gaps corrected
        corrected_waveforms: waveform array with interpolated or filled values
    """

    pass