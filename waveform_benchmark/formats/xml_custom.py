
import math
import os

import jpyutil
import numpy
from dotenv import load_dotenv

from waveform_benchmark.formats.base import BaseFormat

# Load variables from .env file
load_dotenv()


def import_jpy():
    """
    This import the jpy module by first checking to see if the user can successfully call import jpy. If they cannot
    it checks to see if the JPY_JVM_DLL environment variable (pointing to the JVM library) is set. If it is set, call
    try to load it and then import jpy. Otherwise ask the user to set or correct the path to the library.
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


jpy = import_jpy()


class XMLCustom(BaseFormat):
    """
    This class allows for writing and reading the custom XML waveform format
    """

    def write_waveforms(self, path, waveforms):

        # create JVM and get object
        jvm_options = ['-Xmx8G', '-Xms1G', '-Djava.class.path=./waveform_benchmark/formats/xml_java/mimicwf.jar:./waveform_benchmark/formats/xml_java/ccsixml.jar:./waveform_benchmark/formats/xml_java/ccsicore.jar:./waveform_benchmark/formats/xml_java/imslib.jar:./waveform_benchmark/formats/xml_java/xbean.jar:./waveform_benchmark/formats/xml_java/xercesImpl.jar:./waveform_benchmark/formats/xml_java/xml-apis.jar']
        jpy.create_jvm(options=jvm_options)
        WaveForm2XML_class = jpy.get_type('org.tmc.b2ai.importer.waveform.WaveForm2XML')
        obj = WaveForm2XML_class()

        # get length of longest signal
        length = max(waveform['chunks'][-1]['end_time']
                     for waveform in waveforms.values())

        # Convert each channel into an array with no gaps.
        for name, waveform in waveforms.items():
            sig_length = round(length * waveform['samples_per_second'])
            samples = numpy.empty(sig_length, dtype=numpy.float32)
            samples[:] = numpy.nan
            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']
                samples[start:end] = chunk['samples']

            # assign the signal to our object
            obj.setSignal(name, samples, waveform['samples_per_second'], waveform['units'])

        # save the XML file by writing out our object
        obj.writeToXML(path, True)

    def read_waveforms(self, path, start_time, end_time, signal_names):

        # create JVM and get object
        jpy.create_jvm([
                           '-Djava.class.path=./waveform_benchmark/formats/xml_java/mimicwf.jar:./waveform_benchmark/formats/xml_java/ccsixml.jar:./waveform_benchmark/formats/xml_java/ccsicore.jar:./waveform_benchmark/formats/xml_java/imslib.jar:./waveform_benchmark/formats/xml_java/xbean.jar:./waveform_benchmark/formats/xml_java/xercesImpl.jar:./waveform_benchmark/formats/xml_java/xml-apis.jar'])
        WaveForm2XML_class = jpy.get_type('org.tmc.b2ai.importer.waveform.WaveForm2XML')
        obj = WaveForm2XML_class()

        # Extract the requested samples from the array.
        results = {}
        for signal_name in signal_names:
            length = end_time - start_time
            results[signal_name] = numpy.array(obj.getSignal(signal_name, start_time, length, path), dtype='int32')

        return results
