# %%
import waveform_benchmark.formats.dcm_rand_access as dcmra
import os

from typing import BinaryIO, Any, cast
from pydicom.fileutil import PathType

from pydicom.tag import Tag
from pydicom.dataset import Dataset
from pydicom.dataelem import DataElement

# from pydicom.datadict import _dictionary_vr_fast
from pydicom.sequence import Sequence

import numpy as np
from pydicom.waveforms.numpy_handler import WAVEFORM_DTYPES


def get_tag(file_obj : BinaryIO, ds : Dataset, tag : str,
                           defer_size: int = 256) -> DataElement:
    t = Tag(tag)
    raw = ds._dict.get(t)
    
    el = dcmra.partial_read_deferred_data_element(file_obj, raw_data_elem = raw, child_defer_size = defer_size)  # waveform sequence
    return el

# read the sequence object to get metadata (channel name, group, freq, time_offsets, etc)
# return a list of dictionary suitable for conversion to a pandas dataframe
def get_waveform_seq_info(fileobj: BinaryIO,
                          group : DataElement, defer_size: int = 256) -> list:
    channel_info = []
    # collect basic waveform info:  channel name, freq, time offset, number of samples, and group label

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
            f"Unable to convert the waveform multiplex group "
            f"as the following required elements are missing from "
            f"the sequence multiplex_group: {', '.join(missing)}"
        )

    # Determine the expected length of the data (without padding)
    time_offset = cast(float, group.MultiplexGroupTimeOffset) / 1000.0  # convert from millisec to sec.
    nr_channels = cast(int, group.NumberOfWaveformChannels)
    sample_freq = cast(float, group.SamplingFrequency)
    nr_samples = cast(int, group.NumberOfWaveformSamples)
    group_label = cast(str, group.MultiplexGroupLabel)
    padding_value = int.from_bytes( group.WaveformPaddingValue, byteorder="little", signed=True )

    # then load the channel definition sequence ONLY, and fully deserialize it.
    # extract the channel name, and interleave id.
    chdefs = get_tag(fileobj, group, 'ChannelDefinitionSequence', defer_size = 100)
    # print(type(chdefs))
    
    ch_defs = cast(list[Dataset], chdefs)
    for ch_i, ch in enumerate(ch_defs):
        coding = ch.ChannelSourceSequence[0]
        channel_info.append({
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
def get_multiplex_array(fileobj : BinaryIO, 
                       multiplex_group: Dataset, 
                       start_offset: int = 0, end_offset: int = -1, as_raw: bool = True) -> "np.ndarray":
    """Return an :class:`~numpy.ndarray` for the multiplex group in the
    *Waveform Sequence* at `index`.

    .. versionadded:: 2.1

    Parameters
    ----------
    multiplex_group : pydicom.dataset.Dataset
        The :class:`Dataset` containing a :dcm:`Waveform
        <part03/sect_C.10.9.html>` module and the *Waveform Sequence* to be
        converted.
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
    if multiplex_group is None:
        raise AttributeError(
            "No (5400,0100) Waveform Sequence element found in the dataset"
        )

    required_elements = [
        "NumberOfWaveformChannels",
        "NumberOfWaveformSamples",
        "WaveformBitsAllocated",
        "WaveformSampleInterpretation",
        "WaveformPaddingValue",
        "WaveformData",
    ]
    missing = [elem for elem in required_elements if elem not in multiplex_group]
    if missing:
        raise AttributeError(
            f"Unable to convert the waveform multiplex group "
            f"as the following required elements are missing from "
            f"the sequence multiplex_group: {', '.join(missing)}"
        )

    # Determine the expected length of the data (without padding)
    bytes_per_sample = cast(int, multiplex_group.WaveformBitsAllocated) // 8
    nr_samples = cast(int, multiplex_group.NumberOfWaveformSamples)
    nr_channels = cast(int, multiplex_group.NumberOfWaveformChannels)
    bits_allocated = cast(int, multiplex_group.WaveformBitsAllocated)
    sample_interpretation = cast(str, multiplex_group.WaveformSampleInterpretation)
    padding_value = int.from_bytes( multiplex_group.WaveformPaddingValue, byteorder="little", signed=True )
    
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
    wavedata = multiplex_group._dict.get(Tag('WaveformData'))  #54001010

    # read it.
    raw_arr = dcmra.read_range(fileobj, wavedata, start, end)
    
    # note data is transposed because of value interleaving.
    arr = np.frombuffer(cast(bytes, raw_arr), dtype=dtype)
    # Reshape to (samples, channels) and make writeable, and transpose (was interleaved)
    arr = np.copy(arr.reshape(etimestamp - stimestamp, nr_channels).T)

    # If not raw, then apply sensitivity correction
    if not as_raw:
        channel_seqs = get_tag(fileobj, multiplex_group, 'ChannelDefinitionSequence', defer_size = 100) # 003a0200
        chseq = cast(list["Dataset"], channel_seqs)
        
        # Apply correction factor (if possible)
        arr = arr.astype("float")
        for ch_idx, ch in enumerate(chseq):
            baseline = float(ch.ChannelBaseline)
            sensitivity = float(ch.ChannelSensitivity)
            correction = float(ch.ChannelSensitivityCorrectionFactor)
            adjustment = sensitivity * correction
            if (adjustment != 1.0) and (baseline != 0.0):
                arr[ch_idx, ...] = np.where(arr[ch_idx, ...] == padding_value, np.nan, arr[ch_idx, ...] * adjustment + baseline)
            elif (adjustment != 1.0):
                arr[ch_idx, ...] = np.where(arr[ch_idx, ...] == padding_value, np.nan, arr[ch_idx, ...] * adjustment)
            elif (baseline != 0.0):
                arr[ch_idx, ...] = np.where(arr[ch_idx, ...] == padding_value, np.nan, arr[ch_idx, ...] + baseline)
            else:
                arr[ch_idx, ...] = np.where(arr[ch_idx, ...] == padding_value, np.nan, arr[ch_idx, ...])


    return cast("np.ndarray", arr)

