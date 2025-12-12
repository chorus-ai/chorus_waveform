import abc
from typing import Dict, Tuple, Any, List, Array, Literal
from src.qc.gaps import (
    detect_gaps_stream, detect_gaps_pair,
    correct_gaps_stream, correct_gaps_pair
)

class BaseSite(abc.ABC):

    @abc.abstractmethod
    def assemble_waveforms(self, path: str):
        """
        Search directory for waveform file paths.
        For each file path, extract columns:

            'PATIENT_ID': str
            'SESSION_ID': str
            'PROCESS': bool
            'FILE_NAME': str
            'FORMAT':
        
        :param self: object instance
        :param path: directory
        :type path: str
        """
        raise NotImplementedError

    @abc.abstractmethod
    def write_waveforms(self, path: str, waveforms: dict):
        raise NotImplementedError
    
    @abc.abstractmethod
    def read_waveforms(self, path: str, start_time: float, end_time: float,
                       signal_names: list):
        """
        Call on BaseFormat.read().
        Format to standard output for other operations:

            1. quality checks
            2. gap correction
            3. WFDB conversion

        Reference site-specific name and structure irregularities.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_metadata(self, path: str):
        raise NotImplementedError
    
    @abc.abstractmethod
    def gaps(self, data: Array, attrs: Dict):
        """
        Prepare data and attributes for shared tool
        to identify and correct gaps.
        """
        # detect_gaps_pair()
        # detect_gaps_stream()
        # correct_gaps_pair()
        # correct_gaps_stream()
        raise NotImplementedError

class BaseFormat(abc.ABC):
    """
    Abstract class for extracting data and attributes from specific file format.

    For each data format that we want to benchmark, a class must be
    defined that inherits from this class and defines the following
    methods:

    - can_read: flag for system-readable
    - read: prepare file attributes for site-specific read
    - write: prepare file attributes for site-specific write
    """

    @abc.abstractmethod
    def can_read(self, path: str):
        """Return True if this format can read the given file."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def read(self, path: str, start_time: float = None, end_time: float = None, 
             signals: List[str] = None):
        """
        Prepare waveform attributes from file for site-native read commands:

            1. BaseSite.read_waveforms()
            2. BaseSite.get_metadata()

        Returns:
            waveforms: dict mapping signal names to numpy arrays
            metadata: dict of file metadata (sampling rate, channels, units)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def write(self, path: str, waveforms: Dict[str, Any], metadata: Dict[str, Any]):
        """
        Prepare waveform attributes from file for site-native write commands:

            1. BaseSite.write_waveforms()
        """
        raise NotImplementedError


# taken directly from this repo:
# https://github.com/chorus-ai/chorus_waveform/blob/main/waveform_benchmark/formats/base.py

class BaseFormat(abc.ABC):
    """
    Abstract class for data formats to be tested.

    For each data format that we want to benchmark, a class must be
    defined that inherits from this class and defines the following
    methods:

    - write_waveforms: take a collection of waveform data, and save
      the data to one or more files on disk

    - read_waveforms: load waveforms from one or more files, and
      return the requested subset of the data
    """

    @abc.abstractmethod
    def write_waveforms(self, path: str, waveforms: dict):
        raise NotImplementedError

    @abc.abstractmethod
    def read_waveforms(self, path: str, start_time: float, end_time: float,
                       signal_names: list):
        raise NotImplementedError
    
    # kwargs is a dictionary that can be used to pass additional arguments to the format
    # replaces the total_length, block_length, and block_size params which are either test specific or intrinsic to the format.
    def open_waveforms(self, path: str, signal_names:list, **kwargs):
        raise NotImplementedError

    def read_opened_waveforms(self, opened_files: dict, start_time: float, end_time: float,
                             signal_names: list):
        raise NotImplementedError

    def close_waveforms(self, opened_files: dict):
        raise NotImplementedError
      
      
    def open_read_close_waveforms(self, path, start_time, end_time, signal_names, **kwargs):
        opened_files = self.open_waveforms(path, signal_names, **kwargs)
        output = self.read_opened_waveforms(opened_files, start_time, end_time, signal_names)
        self.close_waveforms(opened_files)
        
        return output