
import math
import os

import jpyutil
import numpy
from dotenv import load_dotenv

from waveform_benchmark.formats.base import BaseFormat

# Load variables from .env file
load_dotenv()

# get path to this file, so we can build a path relative to this
current_file_path = os.path.dirname(os.path.abspath(__file__))


def import_jpy():
    """
    This imports the jpy module by first checking to see if `import jpy` can be successfully called. If not, it checks
    to see if the JPY_JVM_DLL environment variable (pointing to the JVM library) is set. If it is set, load
    it and then `import jpy`. Otherwise, ask the user to set or correct the path to the library.
    """
    try:
        import jpy
    except ImportError:
        if 'JPY_JVM_DLL' in os.environ:
            path = os.environ['JPY_JVM_DLL']
            try:
                jpyutil.preload_jvm_dll()
                import jpy
            except UnboundLocalError:
                print(f'The JPY_JVM_DLL variable in your .env file is set to {path}, this may not be correct. Please point it to your JVM libarary')
        else:
            print('Set the JPY_JVM_DLL variable to point to your JVM library in your .env file')
    return jpy


def get_path_to_jar_files():
    """
    build the string to all of the .jar files
    """
    jar_string=""
    files = ['mimicwf.jar', 'ccsixml.jar', 'ccsicore.jar', 'imslib.jar', 'xbean.jar', 'xercesImpl.jar', 'xml-apis.jar']
    for file in files:
        jar_string += os.path.join(current_file_path, 'xml_java', file) + ':'

    # return the string without the extraneous : at the end of the last file
    return jar_string[:-1]


jpy = import_jpy()


class BaseXMLCustom(BaseFormat):
    """
    This class allows for writing and reading the custom XML waveform format
    """

    def write_waveforms(self, path, waveforms):

        # create JVM and get object
        jvm_max_heap = os.getenv('BENCHMARK_XML_JVM_MAX_HEAP', default='-Xmx8G')
        jvm_init_heap = os.getenv('BENCHMARK_XML_JVM_INITIAL_HEAP', default='-Xmx1G')
        jar_path = get_path_to_jar_files()
        jvm_options = [jvm_max_heap, jvm_init_heap, f'-Djava.class.path={jar_path}']
        jpy.create_jvm(options=jvm_options)
        WaveForm2XML_class = jpy.get_type('org.tmc.b2ai.importer.waveform.WaveForm2XML')
        obj = WaveForm2XML_class()

        # get the length of longest signal
        length = max(waveform['chunks'][-1]['end_time']
                     for waveform in waveforms.values())

        # Convert each channel into an array with no gaps.
        for name, waveform in waveforms.items():
            sig_length = round(length * waveform['samples_per_second'])
            samples = numpy.empty(sig_length, dtype=numpy.float32)
            samples[:] = -200000
            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']
                samples[start:end] = chunk['samples']

            # assign the signal to our object
            obj.setSignal(name, samples, waveform['samples_per_second'], waveform['units'])

        # save the XML file by writing out our object
        obj.writeToXML(path, self.compressed)

    def read_waveforms(self, path, start_time, end_time, signal_names):

        # create JVM and get object
        jar_path = get_path_to_jar_files()
        jpy.create_jvm([f'-Djava.class.path={jar_path}'])
        WaveForm2XML_class = jpy.get_type('org.tmc.b2ai.importer.waveform.WaveForm2XML')
        obj = WaveForm2XML_class()

        # Extract the requested samples from the array.
        results = {}
        for signal_name in signal_names:
            length = end_time - start_time
            results[signal_name] = numpy.array(obj.getSignal(signal_name, start_time, length, path), dtype='int32')

        return results


class XMLCustomUncompressed(BaseXMLCustom):
    """
    Don't compress the file
    """
    compressed = False


class XMLCustomCompressed(BaseXMLCustom):
    """
    Compress the file
    """
    compressed = True
