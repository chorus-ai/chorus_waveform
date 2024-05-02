# %%
import waveform_benchmark.formats.dcm_rand_access as dcmra
import os

from typing import BinaryIO, Any, cast
from pydicom.fileutil import PathType
from pydicom import dcmread

from pydicom.tag import Tag
from pydicom.dataset import Dataset
from pydicom.dataelem import DataElement

# from pydicom.datadict import _dictionary_vr_fast
from pydicom.sequence import Sequence

import numpy as np
from pydicom.waveforms.numpy_handler import WAVEFORM_DTYPES


# read from a file and return the waveformsequence list, with deferred read
# TODO: generalize to other tags.  need DataSet and Seq tag name as input
def get_waveform_sequences(fileobj_type: Any, filename_or_obj: PathType | BinaryIO, 
                           defer_size: int = 256) -> DataElement:
    """Return the *Waveform Sequence* from the *Waveform Module*.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The :class:`Dataset` containing the *Waveform Module*.

    Returns
    -------
    pydicom.sequence.Sequence
        The *Waveform Sequence* (5400,0100) from the *Waveform Module*.
    """

    # If it wasn't read from a file, then return an error
    if filename_or_obj is None:
        raise OSError("Deferred read -- original filename not stored. Cannot re-open")

    # first open the file
    if isinstance(filename_or_obj, str):
        if not os.path.exists(filename_or_obj):
            raise OSError(
                f"Deferred read -- original file {filename_or_obj} is missing"
            )

        with fileobj_type(filename_or_obj, "rb") as f:
            ds = dcmread(f, defer_size = defer_size)
    else:
        ds = dcmread(filename_or_obj, defer_size = defer_size)
        
    # next get the waveform sequence RAW Data Element
    seqs_raw = ds._dict.get(Tag('WaveformSequence'))  # 54000100
    
    seqs = dcmra.partial_read_deferred_data_element(fileobj_type, filename_or_obj, timestamp = None, raw_data_elem = seqs_raw, child_defer_size = defer_size)  # waveform sequence
    return seqs


# read the sequence object to get metadata (channel name, group, freq, time_offsets, etc)
# return a list of dictionary suitable for conversion to a pandas dataframe
def get_waveform_seq_info(fileobj_type: Any, filename_or_obj: PathType | BinaryIO, 
                          seqs: "Sequence", defer_size: int = 256) -> list:
    if seqs is None:
        raise AttributeError(
            "No (5400,0100) Waveform Sequence element found in the dataset"
        )

    channel_info = []
    # collect basic waveform info:  channel name, freq, time offset, number of samples, and group label
    for seq_id, group in enumerate(seqs):
        required_elements = [
            "MultiplexGroupTimeOffset",
            "NumberOfWaveformChannels",
            "SamplingFrequency",
            "NumberOfWaveformSamples",
            "MultiplexGroupLabel",
            "WaveformPaddingValue",
            "ChannelDefinitionSequence",
        ]
        missing = [elem for elem in required_elements if elem not in group]
        if missing:
            raise AttributeError(
                f"Unable to convert the waveform multiplex group with index "
                f"{seq_id} as the following required elements are missing from "
                f"the sequence item: {', '.join(missing)}"
            )

        # Determine the expected length of the data (without padding)
        time_offset = cast(float, group.MultiplexGroupTimeOffset) / 1000.0  # convert from millisec to sec.
        nr_channels = cast(int, group.NumberOfWaveformChannels)
        sample_freq = cast(float, group.SamplingFrequency)
        nr_samples = cast(int, group.NumberOfWaveformSamples)
        group_label = cast(str, group.MultiplexGroupLabel)
        padding_value = cast(int, group.WaveformPaddingValue)
        group_idx = seq_id

        # then load the channel definition sequence
        # extract the channel name, and interleave id.
        chdefs_raw = group._dict.get(Tag('ChannelDefinitionSequence'))
        chdefs = dcmra.partial_read_deferred_data_element(fileobj_type, 
                                                          filename_or_obj, 
                                                          timestamp = None, 
                                                          raw_data_elem = chdefs_raw, 
                                                          child_defer_size = defer_size)  # waveform sequence
        
        channel_defs = cast(list[Dataset], chdefs)
        for ch_i, ch in enumerate(channel_defs):
            coding = ch.ChannelSourceSequence[0]
            required_elements2 = [
                "CodeMeaning",
            ]
            missing2 = [elem for elem in required_elements2 if elem not in coding]
            if missing2:
                raise AttributeError(
                    f"Unable to convert the channel definition with index "
                    f"{ch_i} in group {seq_id} as the following required elements are missing from "
                    f"the sequence item: {', '.join(missing2)}"
                )
            
        
            channel_info.append({
                'group_idx': group_idx,
                'group_label': group_label,
                'channel': cast(str, coding.CodeMeaning),
                'channel_idx': ch_i,
                'start_time': time_offset,
                'freq': sample_freq,
                'padding_value': padding_value,
                'number_samples': nr_samples,
                'number_channels': nr_channels
            })
            
    return channel_info


# read from a file and return the specified waveform in sequence at the specified time window.
# modified version of pydicom.waveforms.numpy_handler.get_multiplex_array.
def get_multiplex_array(fileobj_type: Any, filename_or_obj: PathType | BinaryIO, 
                       seqs: "Sequence", seq_id: int, 
                       start_offset: int = 0, end_offset: int = -1, as_raw: bool = True) -> "np.ndarray":
    """Return an :class:`~numpy.ndarray` for the multiplex group in the
    *Waveform Sequence* at `index`.

    .. versionadded:: 2.1

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The :class:`Dataset` containing a :dcm:`Waveform
        <part03/sect_C.10.9.html>` module and the *Waveform Sequence* to be
        converted.
    seq_id : int
        The index of the multiplex group to return.
    start_offset : int, optional
        The starting sample offset to return. 0 indicates the beginning of the array.
    end_offset : int, optional
        the ending sample offset to return.  If -1, then return all samples.
    as_raw : bool, optional
        If ``True`` (default), then return the raw unitless waveform data. If
        ``False`` then attempt to convert the raw data for each channel to the
        quantity specified by the corresponding (003A,0210) *Channel
        Sensitivity* unit.

    Returns
    -------
    np.ndarray
        The waveform data for a multiplex group as an :class:`~numpy.ndarray`
        with shape (samples, channels).
    """
    if seqs is None:
        raise AttributeError(
            "No (5400,0100) Waveform Sequence element found in the dataset"
        )

    item = cast(list["Dataset"], seqs)[seq_id]
    required_elements = [
        "NumberOfWaveformChannels",
        "NumberOfWaveformSamples",
        "WaveformBitsAllocated",
        "WaveformSampleInterpretation",
        "WaveformData",
    ]
    missing = [elem for elem in required_elements if elem not in item]
    if missing:
        raise AttributeError(
            f"Unable to convert the waveform multiplex group with index "
            f"{seq_id} as the following required elements are missing from "
            f"the sequence item: {', '.join(missing)}"
        )

    # Determine the expected length of the data (without padding)
    bytes_per_sample = cast(int, item.WaveformBitsAllocated) // 8
    nr_samples = cast(int, item.NumberOfWaveformSamples)
    nr_channels = cast(int, item.NumberOfWaveformChannels)
    bits_allocated = cast(int, item.WaveformBitsAllocated)
    sample_interpretation = cast(str, item.WaveformSampleInterpretation)
    
    # Waveform Data is ordered as (C = channel, S = sample):
    # C1S1, C2S1, ..., CnS1, C1S2, ..., CnS2, ..., C1Sm, ..., CnSm
    dtype = WAVEFORM_DTYPES[(bits_allocated, sample_interpretation)]
    
    stimestamp = max(0, start_offset)
    etimestamp = min(nr_samples, end_offset)
    
    if (stimestamp >= etimestamp):
        return np.array([])
    
    start = stimestamp * nr_channels * bytes_per_sample
    end = etimestamp * nr_channels * bytes_per_sample

    # get the RawDataElement object
    wavedata = item._dict.get(Tag('WaveformData'))  #54001010

    # read it.
    raw_arr = dcmra.read_range(fileobj_type, filename_or_obj, wavedata, start, end)
    
    # note data is transposed because of value interleaving.
    arr = np.frombuffer(cast(bytes, raw_arr), dtype=dtype)
    # Reshape to (samples, channels) and make writeable, and transpose (was interleaved)
    arr = np.copy(arr.reshape(etimestamp - stimestamp, nr_channels).T)

    # If not raw, then apply sensitivity correction
    if not as_raw:
        cds = item._dict.get(Tag('ChannelDefinitionSequence'))  # 003a0200
        # get the ChannelDefinitionSequence
        
        # need to set child_defer_size to force reading of SQ of known length
        channel_seqs = dcmra.partial_read_deferred_data_element(fileobj_type, filename_or_obj, timestamp = None, raw_data_elem = cds, child_defer_size = 100)  
        
        # Apply correction factor (if possible)
        arr = arr.astype("float")
        chseq = cast(list["Dataset"], channel_seqs)
        for jj, ch in enumerate(chseq):
            baseline = ch.get("ChannelBaseline", 0.0)
            sensitivity = ch.get("ChannelSensitivity", 1.0)
            correction = ch.get("ChannelSensitivityCorrectionFactor", 1.0)
            adjustment = sensitivity * correction
            arr[..., jj] = arr[..., jj] * adjustment + baseline

    return cast("np.ndarray", arr)

