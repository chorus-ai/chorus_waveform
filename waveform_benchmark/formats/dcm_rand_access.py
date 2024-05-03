# %%
import os

import pydicom
import pydicom._storage_sopclass_uids

from typing import BinaryIO, Any, cast
from pydicom.fileutil import (
    PathType,
    read_undefined_length_value,
    _unpack_tag,
)
from pydicom import config
from pydicom.config import logger

from collections.abc import Iterator, Callable, MutableSequence
from pydicom.tag import (
    ItemTag,
    SequenceDelimiterTag,
    Tag,
    BaseTag,
)
from pydicom.charset import default_encoding, convert_encodings
from pydicom.misc import size_in_bytes
from struct import Struct, unpack
from pydicom.valuerep import EXPLICIT_VR_LENGTH_32, VR as VR_
ENCODED_VR = {vr.encode(default_encoding) for vr in VR_}
from pydicom.dataset import Dataset

from pydicom.dataelem import (
    DataElement,
    RawDataElement,
    empty_value_for_VR,
)
# from pydicom.datadict import _dictionary_vr_fast
from pydicom.sequence import Sequence
from pydicom._dicom_dict import DicomDictionary, RepeatersDictionary
from pydicom.datadict import mask_match


#%%
# my version of deferred_data_element reader that does not recursively read deferred elements
# the original version uses the data_element_generator with no defer_size to reload
# the deferred element as RawDataElement, but with everything populated.
# here we want to read the deferred element as DataElement, but limit the size of the children.

# backport from pydicom.util.hexutil
def bytes2hex(byte_array: bytes) -> str:
    """Return a hexadecimal string representation of a bytes object."""
    return bytes.hex()

# backport from pydicom.values (needed for deferred_data_element_generator)
def _dictionary_vr_fast(tag: int) -> str:
    """Return the VR corresponding to `tag`"""
    # Faster implementation of `dictionary_VR`
    try:
        return DicomDictionary[tag][0]
    except KeyError:
        if not tag >> 16 % 2 == 1:
            mask_x = mask_match(tag)
            if mask_x:
                return RepeatersDictionary[mask_x][0]

        raise KeyError(f"Tag {Tag(tag)} not found in DICOM dictionary")


#%%

# trying to convert the deferred element to DataElement.
# however, implicitVR means vr=None, and data_element_generator's 
# conditional statements will check first for defined length, 
# and create a RawDataElement instead of DataElement.
# if undefined length, then it checks for SQ and reads the sequence into a DataElement
# 
# defined length SQ needs to be checked and read as Sequence with DataElement return type.


# modified version of pydicom.filereader.data_element_generator
# if defer_size is None, materialize self.
# if child_defer_size is None, materialize child, else defer children.
def deferred_data_element_generator(
    fp: BinaryIO,
    is_implicit_VR: bool,
    is_little_endian: bool,
    stop_when: Callable[[BaseTag, str | None, int], bool] | None = None,
    defer_size: int | str | float | None = None,
    encoding: str | MutableSequence[str] = default_encoding,
    specific_tags: list[BaseTag | int] | None = None,
    child_defer_size: int | str | float | None = None,
    
) -> Iterator[RawDataElement | DataElement]:
    """Create a generator to efficiently return the raw data elements.

    .. note::

        This function is used internally - usually there is no need to call it
        from user code. To read data from a DICOM file, :func:`dcmread`
        shall be used instead.

    Parameters
    ----------
    fp : file-like
        The file-like to read from.
    is_implicit_VR : bool
        ``True`` if the data is encoded as implicit VR, ``False`` otherwise.
    is_little_endian : bool
        ``True`` if the data is encoded as little endian, ``False`` otherwise.
    stop_when : None, callable, optional
        If ``None`` (default), then the whole file is read. A callable which
        takes tag, VR, length, and returns ``True`` or ``False``. If it
        returns ``True``, ``read_data_element`` will just return.
    defer_size : int, str or float, optional
        See :func:`dcmread` for parameter info.
    encoding : str | MutableSequence[str]
        Encoding scheme
    specific_tags : list or None
        See :func:`dcmread` for parameter info.

    Yields
    -------
    RawDataElement or DataElement
        Yields DataElement for undefined length UN or SQ, RawDataElement
        otherwise.
    """

    # Summary of DICOM standard PS3.5-2008 chapter 7:
    # If Implicit VR, data element is:
    #    tag, 4-byte length, value.
    #        The 4-byte length can be FFFFFFFF (undefined length)*
    #
    # If Explicit VR:
    #    if OB, OW, OF, SQ, UN, or UT:
    #       tag, VR, 2-bytes reserved (both zero), 4-byte length, value
    #           For all but UT, the length can be FFFFFFFF (undefined length)*
    #   else: (any other VR)
    #       tag, VR, (2 byte length), value
    # * for undefined length, a Sequence Delimitation Item marks the end
    #        of the Value Field.
    # Note, except for the special_VRs, both impl and expl VR use 8 bytes;
    #    the special VRs follow the 8 bytes with a 4-byte length

    # With a generator, state is stored, so we can break down
    #    into the individual cases, and not have to check them again for each
    #    data element
    from pydicom.values import convert_string

    endian_chr = "><"[is_little_endian]

    # assign implicit VR struct to variable as use later if VR assumed missing
    implicit_VR_unpack = Struct(f"{endian_chr}HHL").unpack
    if is_implicit_VR:
        element_struct_unpack = implicit_VR_unpack
    else:  # Explicit VR
        # tag, VR, 2-byte length (or 0 if special VRs)
        element_struct_unpack = Struct(f"{endian_chr}HH2sH").unpack
        extra_length_unpack = Struct(f"{endian_chr}L").unpack  # for lookup speed

    # Make local variables so have faster lookup
    fp_read = fp.read
    fp_seek = fp.seek
    fp_tell = fp.tell
    logger_debug = logger.debug
    debugging = config.debugging
    defer_size = size_in_bytes(defer_size)

    tag_set: set[int] = {tag for tag in specific_tags} if specific_tags else set()
    has_tag_set = bool(tag_set)
    if has_tag_set:
        tag_set.add(0x00080005)  # Specific Character Set

    while True:
        # VR: str | None
        # Read tag, VR, length, get ready to read value
        if len(bytes_read := fp_read(8)) < 8:
            return  # at end of file

        if debugging:
            debug_msg = f"{fp.tell() - 8:08x}: {bytes2hex(bytes_read)}"

        if is_implicit_VR:
            # must reset VR each time; could have set last iteration (e.g. SQ)
            vr = None
            group, elem, length = element_struct_unpack(bytes_read)
        else:  # explicit VR
            group, elem, vr, length = element_struct_unpack(bytes_read)
            # defend against switching to implicit VR, some writer do in SQ's
            # issue 1067, issue 1035

            if vr in ENCODED_VR:  # try most likely solution first
                vr = vr.decode(default_encoding)
                if vr in EXPLICIT_VR_LENGTH_32:
                    bytes_read = fp_read(4)
                    length = extra_length_unpack(bytes_read)[0]
                    if debugging:
                        debug_msg += " " + bytes2hex(bytes_read)
            elif not (b"AA" <= vr <= b"ZZ") and config.assume_implicit_vr_switch:
                # invalid VR, must be 2 cap chrs, assume implicit and continue
                if debugging:
                    logger.warning(
                        f"Unknown VR '0x{vr[0]:02x}{vr[1]:02x}' assuming "
                        "implicit VR encoding"
                    )
                vr = None
                group, elem, length = implicit_VR_unpack(bytes_read)
            else:
                # Either an unimplemented VR or implicit VR encoding
                # Note that we treat an unimplemented VR as having a 2-byte
                #   length, but that may not be correct
                vr = vr.decode(default_encoding)
                if debugging:
                    logger.warning(
                        f"Unknown VR '{vr}' assuming explicit VR encoding with "
                        "2-byte length"
                    )

        if debugging:
            debug_msg = f"{debug_msg:<47s}  ({group:04X},{elem:04X})"
            if not is_implicit_VR:
                debug_msg += f" {vr} "
            if length != 0xFFFFFFFF:
                debug_msg += f"Length: {length}"
            else:
                debug_msg += "Length: Undefined length (FFFFFFFF)"
            logger_debug(debug_msg)

        # Positioned to read the value, but may not want to -- check stop_when
        value_tell = fp_tell()
        tag = group << 16 | elem
        if tag == 0xFFFEE00D:
            # The item delimitation item of an undefined length dataset in
            #   a sequence, length is 0
            # If we hit this then we're at the end of the current dataset
            return

        if stop_when is not None:
            # XXX VR may be None here!! Should stop_when just take tag?
            if stop_when(BaseTag(tag), vr, length):
                if debugging:
                    logger_debug(
                        "Reading ended by stop_when callback. "
                        "Rewinding to start of data element."
                    )
                rewind_length = 8
                if not is_implicit_VR and vr in EXPLICIT_VR_LENGTH_32:
                    rewind_length += 4
                fp_seek(value_tell - rewind_length)
                return

        # Reading the value
        # First case (most common): reading a value with a defined length
        if length != 0xFFFFFFFF:
            # don't defer loading of Specific Character Set value as it is
            # needed immediately to get the character encoding for other tags
            if has_tag_set and tag not in tag_set:
                # skip the tag if not in specific tags
                fp_seek(fp_tell() + length)
                continue

            # -- if SQ even if length is not zero, then materialize it as sequence of RawDataElements
            
            # VR UN with undefined length - should never happen
            
            # Try to look up type to see if is a SQ
            # if private tag, won't be able to look it up in dictionary,
            #   in which case just ignore it and read the bytes unless it is
            #   identified as a Sequence
            if vr is None:
                try:
                    vr = _dictionary_vr_fast(tag)
                except KeyError:
                    # Look ahead to see if it consists of items
                    # and is thus a SQ
                    next_tag = _unpack_tag(fp_read(4), endian_chr)
                    # Rewind the file
                    fp_seek(fp_tell() - 4)
                    if next_tag == ItemTag:
                        vr = VR_.SQ

            if vr == VR_.SQ:
                if debugging:
                    logger_debug(
                        f"{fp_tell():08X}: Reading/parsing undefined length sequence"
                    )

                seq = deferred_read_sequence(
                    fp, is_implicit_VR, is_little_endian, length, encoding, 
                    defer_size=child_defer_size
                )
                if has_tag_set and tag not in tag_set:
                    continue

                # print("NOTE: return sequence with known length")
                yield DataElement(
                    BaseTag(tag), vr, seq, value_tell, is_undefined_length=False
                )
            else:

                if (
                    defer_size is not None
                    and length > defer_size
                    and tag != 0x00080005  # charset
                ):
                    # Flag as deferred by setting value to None, and skip bytes
                    value = None
                    if debugging:
                        logger_debug(
                            "Defer size exceeded. Skipping forward to next data element."
                        )
                    fp_seek(fp_tell() + length)
                else:
                    value = (
                        fp_read(length)
                        if length > 0
                        else cast(bytes | None, empty_value_for_VR(vr, raw=True))
                    )
                    if debugging:
                        dotdot = "..." if length > 20 else "   "
                        displayed_value = value[:20] if value else b""
                        logger_debug(
                            "%08x: %-34s %s %r %s"
                            % (
                                value_tell,
                                bytes2hex(displayed_value),
                                dotdot,
                                displayed_value,
                                dotdot,
                            )
                        )

                # If the tag is (0008,0005) Specific Character Set, then store it
                if tag == 0x00080005:
                    # *Specific Character String* is b'' for empty value
                    encoding = convert_string(cast(bytes, value) or b"", is_little_endian)
                    # Store the encoding value in the generator
                    # for use with future elements (SQs)
                    encoding = convert_encodings(encoding)

                # print("NOTE: return data element")
                yield RawDataElement(
                    BaseTag(tag),
                    vr,
                    length,
                    value,
                    value_tell,
                    is_implicit_VR,
                    is_little_endian,
                )

        # Second case: undefined length - must seek to delimiter,
        # unless is SQ type, in which case is easier to parse it, because
        # undefined length SQs and items of undefined lengths can be nested
        # and it would be error-prone to read to the correct outer delimiter
        else:
            # VR UN with undefined length shall be handled as SQ
            # see PS 3.5, section 6.2.2
            if vr == VR_.UN and config.settings.infer_sq_for_un_vr:
                vr = VR_.SQ
            # Try to look up type to see if is a SQ
            # if private tag, won't be able to look it up in dictionary,
            #   in which case just ignore it and read the bytes unless it is
            #   identified as a Sequence
            if vr is None or vr == VR_.UN and config.replace_un_with_known_vr:
                try:
                    vr = _dictionary_vr_fast(tag)
                except KeyError:
                    # Look ahead to see if it consists of items
                    # and is thus a SQ
                    next_tag = _unpack_tag(fp_read(4), endian_chr)
                    # Rewind the file
                    fp_seek(fp_tell() - 4)
                    if next_tag == ItemTag:
                        vr = VR_.SQ

            if vr == VR_.SQ:
                if debugging:
                    logger_debug(
                        f"{fp_tell():08X}: Reading/parsing undefined length sequence"
                    )

                seq = deferred_read_sequence(
                    fp, is_implicit_VR, is_little_endian, length, encoding, 
                    defer_size=child_defer_size
                )
                if has_tag_set and tag not in tag_set:
                    continue

                # print("NOTE: return sequence with unknown length")
                yield DataElement(
                    BaseTag(tag), vr, seq, value_tell, is_undefined_length=True
                )
            else:
                if debugging:
                    logger_debug("Reading undefined length data element")

                value = read_undefined_length_value(
                    fp, is_little_endian, SequenceDelimiterTag, defer_size
                )

                # tags with undefined length are skipped after read
                if has_tag_set and tag not in tag_set:
                    continue

                # print("NOTE: return element with unknown length")
                yield RawDataElement(
                    BaseTag(tag),
                    vr,
                    length,
                    value,
                    value_tell,
                    is_implicit_VR,
                    is_little_endian,
                )

# modified version of pydicom.filereader.read_sequence
def deferred_read_sequence(
    fp: BinaryIO,
    is_implicit_VR: bool,
    is_little_endian: bool,
    bytelength: int,
    encoding: str | MutableSequence[str],
    offset: int = 0,
    defer_size: str | int | float | None = None,
) -> Sequence:
    """Read and return a :class:`~pydicom.sequence.Sequence` -- i.e. a
    :class:`list` of :class:`Datasets<pydicom.dataset.Dataset>`.
    """
    if defer_size is None:
        return pydicom.filereader.read_sequence(
            fp = fp,
            is_implicit_VR = is_implicit_VR,
            is_little_endian = is_little_endian,
            bytelength = bytelength,
            encoding = encoding,
            offset = offset
        )
    
    seq = []  # use builtin list to start for speed, convert to Sequence at end
    is_undefined_length = False
    if bytelength != 0:  # SQ of length 0 possible (PS 3.5-2008 7.5.1a (p.40)
        if bytelength == 0xFFFFFFFF:
            is_undefined_length = True
            bytelength = 0

        fp_tell = fp.tell  # for speed in loop
        fpStart = fp_tell()
        while (not bytelength) or (fp_tell() - fpStart < bytelength):
            file_tell = fp_tell()
            dataset = deferred_read_sequence_item(
                fp, is_implicit_VR, is_little_endian, encoding, offset, 
                defer_size = defer_size
            )
            if dataset is None:  # None is returned if hit Sequence Delimiter
                break

            dataset.file_tell = file_tell + offset
            seq.append(dataset)

    sequence = Sequence(seq)
    sequence.is_undefined_length = is_undefined_length
    return sequence


# modified version of pydicom.filereader.read_sequence_item
def deferred_read_sequence_item(
    fp: BinaryIO,
    is_implicit_VR: bool,
    is_little_endian: bool,
    encoding: str | MutableSequence[str],
    offset: int = 0,
    defer_size: str | int | float | None = None,
    
) -> Dataset | None:
    """Read and return a single :class:`~pydicom.sequence.Sequence` item, i.e.
    a :class:`~pydicom.dataset.Dataset`.
    """
    if defer_size is None:
        return pydicom.filereader.read_sequence_item(
            fp = fp,
            is_implicit_VR = is_implicit_VR,
            is_little_endian = is_little_endian,
            encoding = encoding,
            offset = offset
        )
    
    
    seq_item_tell = fp.tell() + offset
    tag_length_format = "<HHL" if is_little_endian else ">HHL"

    try:
        bytes_read = fp.read(8)
        group, element, length = unpack(tag_length_format, bytes_read)
    except BaseException:
        raise OSError(f"No tag to read at file position {fp.tell() + offset:X}")

    tag = (group, element)
    if tag == SequenceDelimiterTag:  # No more items, time to stop reading
        if config.debugging:
            logger.debug(f"{fp.tell() - 8 + offset:08x}: End of Sequence")
            if length != 0:
                logger.warning(
                    f"Expected 0x00000000 after delimiter, found 0x{length:X}, "
                    f"at position 0x{fp.tell() - 4 + offset:X}"
                )
        return None

    if config.debugging:
        if tag != ItemTag:
            # Flag the incorrect item encoding, will usually raise an
            #   exception afterwards due to the misaligned format
            logger.warning(
                f"Expected sequence item with tag {ItemTag} at file position "
                f"0x{fp.tell() - 4 + offset:X}"
            )
        else:
            logger.debug(
                f"{fp.tell() - 4 + offset:08x}: {bytes2hex(bytes_read)}  "
                "Found Item tag (start of item)"
            )

    if length == 0xFFFFFFFF:
        ds = pydicom.filereader.read_dataset(
            fp,
            is_implicit_VR,
            is_little_endian,
            bytelength=None,
            parent_encoding=encoding,
            at_top_level=False,
            defer_size = defer_size
        )
        ds.is_undefined_length_sequence_item = True
    else:
        ds = pydicom.filereader.read_dataset(
            fp,
            is_implicit_VR,
            is_little_endian,
            length,
            parent_encoding=encoding,
            at_top_level=False,
            defer_size = defer_size            
        )
        ds.is_undefined_length_sequence_item = False

        if config.debugging:
            logger.debug(f"{fp.tell() + offset:08X}: Finished sequence item")

    ds.seq_item_tell = seq_item_tell
    return ds

# %%
# modified version of pydicom.filereader.read_deferred_data_element
def partial_read_deferred_data_element(
    fp: BinaryIO,
    raw_data_elem: RawDataElement,
    child_defer_size: int | str | float | None = None,
) -> DataElement | RawDataElement:
    """Read the previously deferred value from the file into memory
    and return a raw data element.

    .. note:

        This is called internally by pydicom and will normally not be
        needed in user code.

    Parameters
    ----------
    fileobj_type : type
        The type of the original file object.
    filename_or_obj : str or file-like
        The filename of the original file if one exists, or the file-like
        object where the data element persists.
    timestamp : float or None
        The time (as given by stat.st_mtime) the original file has been
        read, if not a file-like.
    raw_data_elem : dataelem.RawDataElement
        The raw data element with no value set.

    Returns
    -------
    dataelem.RawDataElement
        The data element with the value set.

    Raises
    ------
    OSError
        If `filename_or_obj` is ``None``.
    OSError
        If `filename_or_obj` is a filename and the corresponding file does
        not exist.
    ValueError
        If the VR or tag of `raw_data_elem` does not match the read value.
    """
    # if not deferring using pydicom's version.
    # if (child_defer_size is None) or (child_defer_size == 0):
    #     print('using pydicom version of read_deferred_data_element')
    #     return pydicom.filereader.read_deferred_data_element(
    #         fileobj_type = type(fp),
    #         filename_or_obj = fp,
    #         timestamp = None,
    #         raw_data_elem = raw_data_elem
    #     )
    
    
    if config.debugging:
        logger.debug("Reading deferred element %r" % str(raw_data_elem.tag))
    # If it wasn't read from a file, then return an error
    if fp is None:
        raise OSError("Deferred read -- fileobj BinaryIO missing")

    is_implicit_VR = raw_data_elem.is_implicit_VR
    is_little_endian = raw_data_elem.is_little_endian
    offset = pydicom.filereader.data_element_offset_to_value(is_implicit_VR, raw_data_elem.VR)
    # Seek back to the start of the deferred element
    fp.seek(raw_data_elem.value_tell - offset)
    # DO NOT USE THIS.  seems to generate a RawDataElement instead of a sequence when sequence is needed.  Value is loaded but is as byte array.  This is just too unpredictable.
    # if child_defer_size is None:
    #     print("NOTE: using pydiom data element generator")
    #     elem_gen = pydicom.filereader.data_element_generator(
    #         fp = fp,
    #         is_implicit_VR = is_implicit_VR,
    #         is_little_endian = is_little_endian,
    #         defer_size = None,
    #     )
    # else:
    elem_gen = deferred_data_element_generator(
        fp = fp, 
        is_implicit_VR = is_implicit_VR, 
        is_little_endian = is_little_endian, 
        defer_size=None, 
        child_defer_size=child_defer_size
    )

    # Read the data element and check matches what was stored before
    # The first element out of the iterator should be the same type as the
    #   the deferred element == RawDataElement
    # elem = cast(RawDataElement, next(elem_gen))
    elem = next(elem_gen)
    if (elem.VR != raw_data_elem.VR) and (raw_data_elem.VR is not None) and (raw_data_elem.VR != "UN"):
        raise ValueError(
            f"Deferred read VR {elem.VR} does not match original {raw_data_elem.VR}"
        )

    if elem.tag != raw_data_elem.tag:
        raise ValueError(
            f"Deferred read tag {elem.tag!r} does not match "
            f"original {raw_data_elem.tag!r}"
        )

    # Everything is ok, now this object should act like usual DataElement
    return elem


# random read from a byte array in the file
def read_range(fileobj: BinaryIO, 
                data : RawDataElement, start: int, end: int,
                ) -> bytes:
    """Read a chunk of bytes from the file at the specified offset."""
    # If it wasn't read from a file, then return an error
    if fileobj is None:
        raise OSError("Deferred read -- fileobj BinaryIO is missing")

    # grab the bytes   ----------- open file
    fileobj.seek(data.value_tell + start)
    return fileobj.read(end - start)
    

