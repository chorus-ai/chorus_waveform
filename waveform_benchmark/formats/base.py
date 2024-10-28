import abc


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
    @abc.abstractmethod
    def open_waveforms(self, path: str, signal_names:list, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def read_opened_waveforms(self, opened_files: dict, start_time: float, end_time: float,
                             signal_names: list):
        raise NotImplementedError

    @abc.abstractmethod
    def close_waveforms(self, opened_files: dict):
        raise NotImplementedError
      
      
    def open_read_close_waveforms(self, path, start_time, end_time, signal_names, **kwargs):
        opened_files = self.open_waveforms(path, signal_names, **kwargs)
        output = self.read_opened_waveforms(opened_files, start_time, end_time, signal_names)
        self.close_waveforms(opened_files)
        
        return output
