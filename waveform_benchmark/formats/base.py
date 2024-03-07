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
