import os

import numpy as np
from pydicom import dcmwrite, dcmread
from pydicom.dataset import Dataset
from pydicom import uid, _storage_sopclass_uids
from pydicom.waveforms.numpy_handler import WAVEFORM_DTYPES
from typing import TYPE_CHECKING, cast


from waveform_benchmark.formats.base import BaseFormat


# waveforms (dict) contains a waveform record, with keys representing each channel (e.g. ['MLII', 'V5']).
# Each channel contains a dictionary with three keys ('units', 'samples_per_second', 'chunks').
# For example: waveforms['V5'] -> {'units': 'mV', 'samples_per_second': 360, 'chunks': [{'start_time': 0.0, 'end_time': 1805.5555555555557, 'start_sample': 0, 'end_sample': 650000, 'gain': 200.0, 'samples': array([-0.065, -0.065, -0.065, ..., -0.365, -0.335,  0.   ], dtype=float32)}]}

class BaseDICOMFormat(BaseFormat):
    """
    Abstract class for WFDB signal formats.
    """

    # Currently, this class uses a single segment and stores many
    # signals in one signal file.  Using multiple segments and
    # multiple signal files could improve efficiency of storage and
    # per-channel access.
    
    # def validate_waveforms(self, waveforms):
    #     lengths = []
    #     freqs = []
    #     for channel, waveform in waveforms.items():
    #         if 'units' not in waveform:
    #             raise ValueError(f"Waveform '{channel}' is missing 'units' attribute")
    #         if 'samples_per_second' not in waveform:
    #             raise ValueError(f"Waveform '{channel}' is missing 'samples_per_second' attribute")
    #         if 'chunks' not in waveform:
    #             raise ValueError(f"Waveform '{channel}' is missing 'chunks' attribute")
    #         freqs.append(waveform['samples_per_second'])
    #         lengths.append(waveform['chunks'][-1]['end_time'])
    #         for chunk in waveform['chunks']:
    #             if 'start_time' not in chunk:
    #                 raise ValueError(f"Chunk in waveform '{channel}' is missing 'start_time' attribute")
    #             if 'end_time' not in chunk:
    #                 raise ValueError(f"Chunk in waveform '{channel}' is missing 'end_time' attribute")
    #             if 'start_sample' not in chunk:
    #                 raise ValueError(f"Chunk in waveform '{channel}' is missing 'start_sample' attribute")
    #             if 'end_sample' not in chunk:
    #                 raise ValueError(f"Chunk in waveform '{channel}' is missing 'end_sample' attribute")
    #             if 'gain' not in chunk:
    #                 raise ValueError(f"Chunk in waveform '{channel}' is missing 'gain' attribute")
    #             if 'samples' not in chunk:
    #                 raise ValueError(f"Chunk in waveform '{channel}' is missing 'samples' attribute")
    #             if len(chunk['samples']) != chunk['end_sample'] - chunk['start_sample']:
    #                 raise ValueError(f"Chunk in waveform '{channel}' has incorrect number of samples")
    #     # check if all the channels have the same length and frequency
    #     if len(set(lengths)) != 1:
    #         raise ValueError("Waveforms have different lengths")
    #     if len(set(freqs)) != 1:
    #         raise ValueError("Waveforms have different sampling frequencies")
        
    # create the outer container
    # for now make type genericECG.  note that there are multiple other types.
    def make_empty_dcm_dataset(self, waveformType):
        
        # file metadata for fast access.
        fileMeta = Dataset()
        fileMeta.MediaStorageSOPClassUID = uid.GeneralECGWaveformStorage
        fileMeta.MediaStorageSOPInstanceUID = uid.generate_uid()
        fileMeta.TransferSyntaxUID = uid.ExplicitVRLittleEndian

        fileDS = Dataset()
        fileDS.file_meta = fileMeta
        
        # Mandatory.  
        fileDS.SOPInstanceUID = uid.generate_uid()
        ## this should be same as the MediaStorageSOPClassUID
        fileDS.SOPClassUID = fileMeta.MediaStorageSOPClassUID
        fileDS.is_little_endian = True
        fileDS.is_implicit_VR = True

        # General Study Module
        fileDS.StudyInstanceUID = uid.generate_uid()
        fileDS.SeriesInstanceUID = uid.generate_uid()
        return fileDS

    def set_order_information(self, dataset, referringPhysicianName = "Unknown", accessionNumber = "0000"):
        # General Study Module
        dataset.ReferringPhysicianName = referringPhysicianName
        dataset.AccessionNumber = accessionNumber
        return dataset

    def set_study_information(self, dataset, 
                              patientID = "N/A", 
                              patientName = "Doe^John",
                              patientBirthDate = "19800101",
                              patientSex = "F"):
        # patient module:
        dataset.PatientID = patientID
        dataset.PatientName = patientName
        dataset.PatientBirthDate = patientBirthDate
        dataset.PatientSex = patientSex
        
        # needed to build DICOMDIR
        dataset.StudyDate = "20200101"
        dataset.StudyTime = "000000"
        dataset.StudyID = "0"
        dataset.Modality = "ECG"
        dataset.SeriesNumber = 0
                
        return dataset
    
    def set_waveform_acquisition_info(self, dataset, 
                                      manufacturer = "Unknown",
                                      modelName = "Unknown"):
        # General Equipment Module
        dataset.Manufacturer = manufacturer
        
        # Waveform Identification Module
        dataset.InstanceNumber = 1
        dataset.ContentDate = "20200101"
        dataset.ContentTime = "040000"
        dataset.AcquisitionDateTime = "20200101040000"
        
        # Acquisition Context Module
        dataset.AcquisitionContextSequence = [Dataset()]
        acqcontext = dataset.AcquisitionContextSequence[0]
        acqcontext.ValueType = "CODE"
        acqcontext.ConceptNameCodeSequence = [Dataset()]
        codesequence = acqcontext.ConceptNameCodeSequence[0]
        codesequence.CodeValue = "113014"
        codesequence.CodingSchemeDesignator = "DCM"
        codesequence.CodingSchemeVersion = '01'
        codesequence.CodeMeaning = "Resting"

        acqcontext.ConceptCodeSequence = [Dataset()]
        codesequence = acqcontext.ConceptCodeSequence[0]
        codesequence.CodeValue = "113014"
        codesequence.CodingSchemeDesignator = "DCM"
        codesequence.CodingSchemeVersion = '01'
        codesequence.CodeMeaning = "Resting"

        return dataset
    
    
    # channels is a list of channels to work with
    # channel_curr_chunk tracks the current chunk for each channel.
    def create_waveform_chunk(self, waveforms, channels, channel_curr_chunk, channel_total_chunks):
                
        # identify the chunks we still need to process.
        unprocessed_chunkid = { channel: channel_curr_chunk[channel] for channel in channels if channel_curr_chunk[channel] < channel_total_chunks[channel] }
        
        # get the start time and duration.
        unprocessed_chunkstart = { channel: waveforms[channel]['chunks'][chunk_id]['start_sample'] for channel, chunk_id in unprocessed_chunkid }
        # find the channels with minimum start time
        multiplex_chunk_channels = [ channel for channel, start in unprocessed_chunkstart.items() if start == min(unprocessed_chunkstart.values()) ]
        
        # ASSUMPTION:  chunks are uniform across channels.
        
        multiplex_chunk_starttimes = [ waveforms[channel]['chunks'][unprocessed_chunkid[channel]]['start_time'] for channel in multiplex_chunk_channels ]
        if len(set(multiplex_chunk_starttimes)) != 1:
            raise ValueError("Chunks have different starting times")
        
        multiplex_chunk_lengths = [ (waveforms[channel]['chunks'][unprocessed_chunkid[channel]]['end_sample'] - waveforms[channel]['chunks'][unprocessed_chunkid[channel]]['start_sample']) for channel in multiplex_chunk_channels ]
        # test that the chunk lengths are all the same
        if len(set(multiplex_chunk_lengths)) != 1:
            raise ValueError("Chunks have different lengths")

        multiplex_chunk_freqs = [ waveforms[channel]['samples_per_second'] for channel in multiplex_chunk_channels ]
        # test that the frequencies are all the same
        if len(set(multiplex_chunk_freqs)) != 1:
            raise ValueError("Chunks have different frequencies")
        
        # create a new multiplex group
        
        # waveform module.  see https://dicom.innolitics.com/ciods/general-ecg/waveform/54000100/003a0200/003a0319
        # shared for multiple waveforms.
        
        # endtime is implicit: start_time + number of samples / sampling frequency
        
        wfDS = Dataset()  # this is a multiplex group...
        # in milliseconds.
        wfDS.MultiplexGroupTimeOffset = 1000 * multiplex_chunk_starttimes[0]
        # wfDS.TriggerTimeOffset = '0.0'
        wfDS.WaveformOriginality = "ORIGINAL"
        wfDS.NumberOfWaveformChannels = len(multiplex_chunk_channels)
        wfDS.SamplingFrequency = multiplex_chunk_freqs[0]
        wfDS.NumberOfWaveformSamples = multiplex_chunk_lengths[0]
        wfDS.MultiplexGroupLabel = str(wfDS.SamplingFrequency) + "Hz"
        wfDS.WaveformBitsAllocated = self.WaveformBitsAllocated  
        wfDS.WaveformSampleInterpretation = self.WaveformBitsAllowed
        
        # channel definitions
        wfDS.ChannelDefinitionSequence = [ Dataset() for i in range(len(multiplex_chunk_channels)) ]
        samples = []
        # now collect the chunks into an array, generate the multiplex group, and increment the chunk ids.
        for i, channel in enumerate(channels):
            if (channel not in multiplex_chunk_channels):
                continue
            
            # extract the info for the chunk
            chunk_id = unprocessed_chunkid[channel]
            unprocessed_chunkid[channel] = unprocessed_chunkid[channel] + 1
            
            chunk = waveforms[channel]['chunks'][chunk_id]
            # start_time = chunk['start_time']  # in multiplex group time offset
            # end_time = chunk['end_time']  # implicit
            # start_sample = chunk['start_sample']  # may be used for channel offset or skew, but here we assume all chunks start together.
            # end_sample = chunk['end_sample']  # not used. assumes all channels have same length.
            
            # TODO: QUANTIZE: need to deal with nan....
            data = np.round(np.frombuffer(chunk['samples']) * float(chunk['gain']), decimals=0).astype(np.int32)
            samples.append(data)  
            unit = waveforms[channel]['units']
            
            # create the channel
            chdef = wfDS.ChannelDefinitionSequence[i]
            # chdef.ChannelTimeSkew = '0'  # Time Skew OR Sample Skew
            chdef.ChannelSampleSkew = "0"
            # chdef.ChannelOffset = '0'
            chdef.WaveformBitsStored = self.WaveformBitsAllocated
            chdef.ChannelSourceSequence = [Dataset()]
            source = chdef.ChannelSourceSequence[0]
            
            # this needs a look up from a controlled vocab.  This is not correct here..
            source.CodeValue = "5.6.3-9-" + str(i+1)
            source.CodingSchemeDesignator = "SCPECG"
            source.CodingSchemeVersion = "1.3"
            source.CodeMeaning = channel
            
            chdef.ChannelSensitivity = 1.0
            chdef.ChannelSensitivityUnitsSequence = [Dataset()]
            units = chdef.ChannelSensitivityUnitsSequence[0]
            
            # this also needs a look up from a controlled vocab.  This is not correct here...
            units.CodeValue = unit
            units.CodingSchemeDesignator = "UCUM"
            units.CodingSchemeVersion = "1.4"
            units.CodeMeaning = "millivolt"  # this needs to be fixed.
            
            # multiplier to apply to the encoded value to get back the orginal input.
            chdef.ChannelSensitivityCorrectionFactor = 1.0 / float(chunk['gain'])
            chdef.ChannelBaseline = '0'
            chdef.WaveformBitsStored = self.WaveformBitsAllocated
            # only for amplifier type of AC
            # chdef.FilterLowFrequency = '0.05'
            # chdef.FilterHighFrequency = '300'
            
            

        # actual bytes. arr is a numpy array of shape np.stack((ch1, ch2,...), axis=1)
        arr = np.stack(samples, axis=1)
        wfDS.WaveformData = arr.tobytes()

        return wfDS, channel_curr_chunk
    
    def add_waveforms(self, dataset, waveforms):
        
        # dicom waveform -> sequence of multiplex group -> channels wiht same sampling frequency
        # multiplex group can have labels.
        # 
        
        # input here:  each channel can have different sampling rate.  each with chunks
        # group sample frequencies together
        # then into chunks.
        
        # first group waveforms by sampling frequency
        channel_freqs = { channel: waveforms[channel]['samples_per_second'] for channel in waveforms.keys() }
        unique_freqs = set([ freq for _, freq in channel_freqs.items() ])
        # invert the channel_freqs dictionary.
        freq_channels = { freq: [ channel for channel, f in channel_freqs.items() if f == freq ] for freq in unique_freqs }
        # we will iterate over these.
        
        
        # this is set up to handle multiple chunks, assuming 
        #   1. channels for a chunk all have the same start and end time, thus the same length.
        channel_total_chunks = { channel: len(waveforms[channel]['chunks']) for channel in waveforms.keys() }
        channel_curr_chunk = { channel: 0 for channel in waveforms.keys() }
        
        dataset.WaveformSequence = []
        # iterate over frequencies
        for _, channels in freq_channels.item():
            
            # iterate over chunks
            while any([ channel_curr_chunk[channel] < channel_total_chunks[channel] for channel in channels ]):
                multiplexGroupDS, channel_curr_chunk = self.create_waveform_chunk(waveforms, channels, 
                                                                                channel_curr_chunk, channel_total_chunks)

                dataset.WaveformSequence.append(multiplexGroupDS)
        
        
        # each unique frequency will have a different label for the multiplex group..  The multiplex group will 
        # have a separate instance contain the channels with the same frequency
        # each chunk will have channels with the same frequency (with same label).  Channels for each multiplex group
        # will have the same start and end time/ sample ids.
        
        return dataset
    
    
    def write_waveforms(self, path, waveforms):
        
        # validate_waveforms(waveforms)
        dicom = Dataset()
        
        # Create DICOM object
        dicom = self.make_empty_dcm_dataset("genericECG")
        dicom = self.set_order_information(dicom)
        dicom = self.set_study_information(dicom)
        dicom = self.set_waveform_acquisition_info(dicom)
        dicom = self.add_waveforms(dicom, waveforms)        
        
        # Save DICOM file.  write_like_original is required
        dicom.save_as(path, write_like_original=False)
        
    
    def read_waveforms(self, path, start_time, end_time, signal_names):

        dicom = dcmread(path)
        
        results = { name: {} for name in signal_names }
        
        for multiplex_group in dicom.WaveformSequence:
            # check match by channel name, start and end time
            dtype = WAVEFORM_DTYPES[(self.WaveformBitsAllocated, self.WaveformSampleInterpretation)]
            
            start_t = multiplex_group.MultiplexGroupTimeOffset / 1000.0
            end_t = start_t + float(multiplex_group.NumberOfWaveformSamples) / float(multiplex_group.SamplingFrequency)

            # out of bounds.  so exclude.
            if (start_t >= end_time) or (end_t <= start_time):
                continue
            
            # inbound.  compute the time:
            chunk_start_t = max(start_t, start_time)
            chunk_end_t = min(end_t, end_time)
            
            # compute the global and chunk sample offsets.
            if (chunk_start_t == start_t):
                chunk_start_sample = 0
                global_start_sample = np.round(start_t * float(multiplex_group.SamplingFrequency))
            else: 
                chunk_start_sample = np.round((chunk_start_t - start_t) * float(multiplex_group.SamplingFrequency))
                global_start_sample = np.round(chunk_start_t * float(multiplex_group.SamplingFrequency))
            if (chunk_end_t == end_t):
                chunk_end_sample = multiplex_group.NumberOfWaveformSamples
                global_end_sample = global_start_sample + chunk_end_sample
            else:
                chunk_end_sample = np.round((chunk_end_t - start_t) * float(multiplex_group.SamplingFrequency))
                global_end_sample = global_start_sample + chunk_end_sample
            
            arr = np.frombuffer(cast(bytes, multiplex_group.WaveformData), dtype=dtype)[chunk_start_sample:chunk_end_sample, :].astype(np.float32)
            for i, channel_def in enumerate(multiplex_group.ChannelDefinitionSequence):
                name = channel_def.ChannelSourceSequence[0].CodeMeaning
                if name not in signal_names:
                    continue

                if name not in results.keys():
                    # results[name] = {}
                    # results[name]['chunks'] = []
                    results[name] = []

                unit = channel_def.ChannelSensitivityUnitsSequence[0].CodeValue
                gain = 1.0 / channel_def.ChannelSensitivityCorrectionFactor
                # results[name]['units'] = unit
                # results[name]['samples_per_second'] = multiplex_group.SamplingFrequency
                arr_i = arr[:, i] * float(channel_def.ChannelSensitivityCorrectionFactor)
                # chunk = {'start_time': chunk_start_t, 'end_time': chunk_end_t, 
                #                              'start_sample': global_start_sample, 'end_sample': global_end_sample,
                #                              'gain': gain, 'samples': arr_i}
                # results[name]['chunks'].append(chunk)
                results[name].append(arr_i)
                
        for name in results.key():
            results[name] = np.concatenate(results[name])

        return results



class DICOMFormat32(BaseDICOMFormat):
    WaveformSampleInterpretation = 'SL'
    WaveformBitsAllocated = 32
    WaveformSampleValueRepresentation = 'SL'
    
class DICOMFormat16(BaseDICOMFormat):
    WaveformSampleInterpretation = 'SS'
    WaveformBitsAllocated = 16
    WaveformSampleValueRepresentation = 'SS'