import time
from typing import cast
import warnings
warnings.filterwarnings("error")

import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom import uid
from pydicom.waveforms.numpy_handler import WAVEFORM_DTYPES

from waveform_benchmark.formats.base import BaseFormat


# types of waveforms and constraints:  
#   https://dicom.nema.org/medical/dicom/current/output/chtml/part03/PS3.3.html
#   https://dicom.nema.org/medical/dicom/current/output/chtml/part17/chapter_C.html  (data organization, and use cases)
#   https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.34.html  (has constraints.)
#   https://dicom.nema.org/medical/dicom/current/output/chtml/part03/chapter_F.html  (dicom dir)
#   https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_F.5.24.html (dicom dir, waveform)
#   https://dicom.nema.org/medical/dicom/current/output/chtml/part03/chapter_A.html#table_A.1-8  (waveform types and IOD modules requirements)

# I, II, III, V, aVR are part of general, 32bit, or 12 lead ECG
# pleth is part of arterial pulse oximetry
# resp is part of respiration.   We need 3 files to handle this.

# the max number of sequence element means that we may not be able to chunk within a file?
# 12 lead ECG is limited to 15 secs.
# sleep EEG is not limited...

# TODO: 1. the list of channels need to be partitioned to multiple files
#   2. multiple series need to be created, each contains a single modality.
#   3. a series then contains a single chunk potentially (e.g. 12 lead ecg).
#   4. so a longer length of data will be split into multiple files in the same series.
#   5. this means files in a series will need to be opened for random access.

# relevant definitions:
# name: modality name, maximum sequences, grouping, SOPUID, max samples, sampling frequency, source, datatype, 
# TwelveLeadECG:  ECG, 1-5, {1: I,II,III; 2: aVR, aVL, aVF; 3: V1, V2, V3; 4: V4, V5, V6; 5: II}, 16384, 200-1000, DCID 3001 “ECG Lead”, SS
# GeneralECGWaveform: ECG, 1-4, 1-24 per serquence, ?, 200-1000, DCID 3001 “ECG Lead”, SS
# General32BitECGWaveform: ECG, 1-4, 1-24 per serquence, ?, by confirmance statement, DCID 3001 “ECG Lead”, SL
# AmbulatoryECGWaveform:  ECG, 1, 1-12, maxsize of wvaeform data attribute, 50-1000, DCID 3001 “ECG Lead”, SB/SS
# HemodynamicWaveform: HD, 1-4, 1-8, maxsize of wvaeform data attribute, <400, , SS
# BasicCardiacElectrophysiologyWaveform: EPS, 1-4, , <=20000, DCID 3011 “Electrophysiology Anatomic Location” , SS
# ArterialPulseWaveform: HD, 1, 1, ? , <600, DCID 3004 “Arterial Pulse Waveform” , SB/SS
# RespiratoryWaveform: RESP, 1, 1, ? , <100, DCID 3005 “Respiration Waveform”, SB/SS
# ScalpEEGWaveform:  EEG, 1 (interruption as separate instances), 1-64, , unconstrained, DCID 3030 “EEG Lead”, SS/SL
# ElectromyogramWaveform: EMG, unconstrained, 1-64, , unconstrained, DCID 3031 “Lead Location Near or in Muscle” or DCID 3032 “Lead Location Near Peripheral Nerve”, SS/SL
# SleepEEGWaveform: EEG, unconstrained, 1-64, , unconstrained, DCID 3030 “EEG Lead” , SS/SL
# MultichannelRespiratoryWaveform: RESP, >1, , unconstrained, DCID 3005 “Respiration Waveform” , SS/SL

# IMPORTANT: look at the specification constraints for different types of waveforms.

# waveforms (dict) contains a waveform record, with keys representing each channel (e.g. ['MLII', 'V5']).
# Each channel contains a dictionary with three keys ('units', 'samples_per_second', 'chunks').
# For example: waveforms['V5'] -> {'units': 'mV', 'samples_per_second': 360, 'chunks': [{'start_time': 0.0, 'end_time': 1805.5555555555557, 'start_sample': 0, 'end_sample': 650000, 'gain': 200.0, 'samples': array([-0.065, -0.065, -0.065, ..., -0.365, -0.335,  0.   ], dtype=float32)}]}

# input waveform assumptions:  a channel will always have same unit and sampling frequency (float).  the gain may change per chunk.  chunks need not be uniform in size or start or end at the same time.
# dicom assumptions:  a mulitplex group contains the multiple channels with the same starting time, frequency and number of samples.  start sample, end time, and end sample are implicit for all channels in the group.
#   each channel can have own unit.  multiplex group can be considered a chunk + channel/frequency unit (????  not sure if this is the best interpretation)
# Our organization:  choices:
#   1. each multiplex group be a chunk+channel
#   2. chunk+frequency (multiple channels)?   If second, we need to pad the short channels, and also some multiplex groups would have more channels (from missing chunks.  harder to search)
#   3. each channel is its own file:  probably like 1. but without the channel mixture so could be faster for 1 channel access. prefer single file, though
#   4. each wave segment is its own file.  1 and 2 still valid questions.  would speed up random access.
#   choose 1.  waveform sequence ordered by time.  compatible with 4.

# TODO: may need to separate channels that are not ECGs in other files.
# TODO: private tags for speeding up metadata access while still maintain compatibility?  this would provide "what do we have", and not necessarily "where is it in the file"
# TODO: compression - in transfer syntax?  likely not standards compliant.
# TODO: random access support?
# TODO: fix the nans...


class BaseDICOMFormat(BaseFormat):
    """
    Abstract class for WFDB signal formats.
    """
    # waveform lead names to dicom IOD mapping. Incomplete.
    # avoiding 12 lead ECG because of the limit in number of samples.
    CHANNEL_TO_DICOM_SPEC = {
        "I": {'IOD': 'General32BitECG',
              'Modality': 'ECG', 'value': '5.6.3-9-1',
              'meaning': 'Lead I (Einthoven)'},
        "II": {'IOD': 'General32BitECG',
               'Modality': 'ECG',
               'value': '5.6.3-9-2',
               'meaning': 'Lead II'},
        "III": {'IOD': 'General32BitECG',
                'Modality': 'ECG',
                'value': '5.6.3-9-61',
                'meaning': 'Lead III'},
        "V": {'IOD': 'General32BitECG',
              'Modality': 'ECG',
              'value': '5.6.3-9-3',
              'meaning': 'Lead V1'},
        "aVR": {'IOD': 'General32BitECG',
                'Modality': 'ECG',
                'value': '5.6.3-9-62',
                'meaning': 'Lead aVR'},
        "Pleth": {'IOD': 'ArterialPulseWaveform',
                  'Modality': 'HD',
                  'value': '5.6.3-9-00',
                  'meaning': 'Plethysmogram'},
        "Resp": {'IOD': 'RespiratoryWaveform',
                 'Modality': 'RESP',
                 'value': '5.6.3-9-01',
                 'meaning': 'Respiration'}
    }

    UCUM_ENCODING = {
        "mV": "millivolt",
        "bpm": "beats per minute",
        "mmHg": "millimeter of mercury",
        "uV": "microvolt",
        "NU": "number",
        "Ohm": "ohm"
    }

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
        fileMeta = FileMetaDataset()
        fileMeta.MediaStorageSOPClassUID = uid.GeneralECGWaveformStorage
        fileMeta.MediaStorageSOPInstanceUID = uid.generate_uid()
        fileMeta.TransferSyntaxUID = uid.ExplicitVRLittleEndian

        fileDS = Dataset()
        fileDS.file_meta = fileMeta

        # Mandatory.  
        fileDS.SOPInstanceUID = uid.generate_uid()
        # this should be same as the MediaStorageSOPClassUID
        fileDS.SOPClassUID = fileMeta.MediaStorageSOPClassUID
        fileDS.is_little_endian = True
        fileDS.is_implicit_VR = True

        # General Study Module
        fileDS.StudyInstanceUID = uid.generate_uid()
        fileDS.SeriesInstanceUID = uid.generate_uid()
        return fileDS

    def set_order_information(self,
                              dataset,
                              referringPhysicianName="Halpert^Jim",
                              accessionNumber="0000"):
        # General Study Module
        dataset.ReferringPhysicianName = referringPhysicianName
        dataset.AccessionNumber = accessionNumber
        return dataset

    def set_study_information(self,
                              dataset,
                              patientID="N/A",
                              patientName="Doe^John",
                              patientBirthDate="19800101",
                              patientSex="F"):
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

    def set_waveform_acquisition_info(self,
                                      dataset,
                                      manufacturer="Unknown",
                                      modelName="Unknown"):
        # General Equipment Module
        dataset.Manufacturer = manufacturer

        # Synchronization Module
        dataset.AcquisitionTimeSynchronized = 'Y'
        dataset.SynchronizationTrigger = "NO TRIGGER"
        #UTC. https://dicom.innolitics.com/ciods/general-ecg/synchronization/00200200
        dataset.SynchronizationFrameOfReferenceUID = "1.2.840.10008.15.1.1"

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
    # channel_chunks is a dict of channel to an array of one or more chunk ids.
    def create_multiplexed_chunk(self, waveforms: dict, channel_chunk: dict):

        for channel in channel_chunk.keys():
            if channel not in self.CHANNEL_TO_DICOM_SPEC.keys():
                raise ValueError("Channel not in CHANNEL_TO_DICOM_SPEC")

        # verify that all chunks are in bound
        if any([i >= len(waveforms[channel]['chunks']) for channel,
                chunks in channel_chunk.items() for i in chunks]):
            raise ValueError("Channel chunk out of bounds:" + channel_chunk)

        multiplex_chunk_freqs = {channel: waveforms[channel]['samples_per_second'] for channel in channel_chunk.keys()}

        # test that the frequencies are all the same
        if len(set(multiplex_chunk_freqs.values())) != 1:
            print("ERROR: Chunks with different freqs", multiplex_chunk_freqs)
            raise ValueError("Chunks have different frequencies")

        unprocessed_chunks = []
        for channel, chunkids in channel_chunk.items():
            chunks = waveforms[channel]['chunks']
            for i in chunkids:
                unprocessed_chunks.append((channel, i, chunks[i]['start_time'],
                                           chunks[i]['start_sample'], chunks[i]['end_sample']))

        min_start_t = min([start_time for _, _, start_time, _, _ in unprocessed_chunks])
        min_start = min([start_sample for _, _, _, start_sample, _ in unprocessed_chunks])
        max_end = max([max_sample for _, _, _, _, max_sample in unprocessed_chunks])
        chunk_max_len = max_end - min_start

        # create a new multiplex group

        # waveform module.  see https://dicom.innolitics.com/ciods/general-ecg/waveform/54000100/003a0200/003a0319
        # shared for multiple waveforms.

        # endtime is implicit: start_time + number of samples / sampling frequency

        wfDS = Dataset()  # this is a multiplex group...
        # in milliseconds.  can only be 16 digits long.  1 year has 31565000000 milliseconds.  we can leave 4 decimal places. and still be within 16 digitas and can handle 3 years.
        wfDS.MultiplexGroupTimeOffset = str(np.round(1000.0 * min_start_t, decimals=4) )  # first start point
        # wfDS.TriggerTimeOffset = '0.0'
        wfDS.WaveformOriginality = "ORIGINAL"
        wfDS.NumberOfWaveformChannels = len(channel_chunk.keys())
        wfDS.SamplingFrequency = next(iter(multiplex_chunk_freqs.values()))
        wfDS.NumberOfWaveformSamples = int(chunk_max_len)
        wfDS.MultiplexGroupLabel = "|".join(channel_chunk.keys())
        wfDS.WaveformBitsAllocated = self.WaveformBitsAllocated
        wfDS.WaveformSampleInterpretation = self.WaveformSampleInterpretation
        wfDS.WaveformPaddingValue = self.PaddingValue.to_bytes(4, 'little', signed=True)

        # channel definitions
        # channel def and samples should be in the same order.
        channeldefs = {}
        samples = {}
        # now collect the chunks into an array, generate the multiplex group, and increment the chunk ids.
        unprocessed_chunks.sort(key=lambda x: (x[0], x[3], x[4])) # ordered by channel
        for channel, chunk_id, start_t, start_s, end_s in unprocessed_chunks:

            chunk = waveforms[channel]['chunks'][chunk_id]
            duration = end_s - start_s
            # start_time = chunk['start_time']  # in multiplex group time offset
            # end_time = chunk['end_time']  # implicit
            # start_sample = chunk['start_sample']  # may be used for channel offset or skew, but here we assume all chunks start together.
            # end_sample = chunk['end_sample']  # not used. assumes all channels have same length.

            # QUANTIZE: and pad missing data
            values = np.nan_to_num(np.frombuffer(chunk['samples'], dtype=np.dtype(chunk['samples'].dtype)), 
                                   nan = float(self.PaddingValue) / float(chunk['gain']) )
            if (duration < chunk_max_len):
                if channel not in samples.keys():
                    samples[channel] = np.array([self.PaddingValue] * chunk_max_len, dtype=self.FileDatatype)

                first = start_s - min_start
                last = end_s - min_start
                samples[channel][first:last] = np.round(values * float(chunk['gain']), decimals=0).astype(self.FileDatatype)
            else:
                samples[channel] = np.round(values * float(chunk['gain']), decimals=0).astype(self.FileDatatype)

            unit = waveforms[channel]['units']

            # create the channel
            chdef = Dataset()
            # chdef.ChannelTimeSkew = '0'  # Time Skew OR Sample Skew
            chdef.ChannelSampleSkew = "0"
            # chdef.ChannelOffset = '0'
            chdef.WaveformBitsStored = self.WaveformBitsAllocated
            chdef.ChannelSourceSequence = [Dataset()]
            source = chdef.ChannelSourceSequence[0]

            # this needs a look up from a controlled vocab.  This is not correct here..
            source.CodeValue = self.CHANNEL_TO_DICOM_SPEC[channel]['value']
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
            units.CodeMeaning = self.UCUM_ENCODING[unit]  # this needs to be fixed.

            # multiplier to apply to the encoded value to get back the orginal input.
            ds = str(1.0 / float(chunk['gain']))
            chdef.ChannelSensitivityCorrectionFactor = ds if len(ds) <= 16 else ds[:16]
            chdef.ChannelBaseline = '0'
            chdef.WaveformBitsStored = self.WaveformBitsAllocated
            # only for amplifier type of AC
            # chdef.FilterLowFrequency = '0.05'
            # chdef.FilterHighFrequency = '300'
            channeldefs[channel] = chdef

        wfDS.ChannelDefinitionSequence = [channeldefs[channel] for channel in channel_chunk.keys()] 
        # actual bytes. arr is a numpy array of shape np.stack((ch1, ch2,...), axis=1)
        arr = np.stack([ samples[channel] for channel in channel_chunk.keys()], axis=1)
        wfDS.WaveformData = arr.tobytes()

        return wfDS

    def add_waveform_chunks_multiplexed(self, dataset, waveforms):
        # dicom waveform -> sequence of multiplex group -> channels with same sampling frequency
        # multiplex group can have labels.

        # input here:  each channel can have different sampling rate.  each with chunks
        # group sample frequencies together
        # then into chunks.

        # first group waveforms by sampling frequency
        channel_freqs = {channel: waveforms[channel]['samples_per_second'] for channel in waveforms.keys()}
        unique_freqs = set([freq for _, freq in channel_freqs.items()])
        # invert the channel_freqs dictionary.
        freq_channels = {freq: [channel for channel,
                                f in channel_freqs.items() if f == freq] for freq in unique_freqs}
        # we will iterate over these.
        print("channels grouped by freq",  freq_channels)

        # we now need to check for each frequency, group the chunks
        freq_ch_chunks = {}
        for freq, channels in freq_channels.items():
            if freq not in freq_ch_chunks.keys():
                freq_ch_chunks[freq] = []

            # get tuples of channel, start/end, channel, and chunkid
            chunk_startend = []
            for c, channel in enumerate(channels):
                chunks = waveforms[channel]['chunks']
                chunk_startend.append([(chunks[i]['start_sample'], c+1, i+1) for i in range(len(chunks))])
                # if start and end times are same for different chunks, "end" comes before "start"
                chunk_startend.append([(chunks[i]['end_sample'], -c-1, -i-1) for i in range(len(chunks))])

            # flatten the list
            chunk_startend = [item for sublist in chunk_startend for item in sublist]
            # sort by time, and c, then i\
            chunk_startend.sort(key=lambda x: (x[0], x[1], x[2]))
            print("Chunk start and end samples", chunk_startend)

            # now identify the grouping of channel-chunks.
            # treat the start and end as parenthesis,
            # we are looking for a nesting that is complete.((())())
            # now we use a stack to identify nesting.
            chunk_stack = set()  # use set for quick match
            channel_chunk = {}
            for time, c, i in chunk_startend:
                if c < 0:
                    chunk_stack.remove((-c, -i))
                    # if empty, then we have a complete group.
                    if len(chunk_stack) == 0:
                        freq_ch_chunks[freq].append(channel_chunk.copy())
                        channel_chunk = {}
                else:
                    chunk_stack.add((c, i))
                    if channels[c-1] not in channel_chunk.keys():
                        channel_chunk[channels[c-1]] = []
                    channel_chunk[channels[c-1]].append(i-1)
            print("Ch and Chunks to process", freq_ch_chunks[freq])

        # process the channel chunks
        dataset.WaveformSequence = []
        for freq, channel_chunks in freq_ch_chunks.items():
            for channel_chunk in channel_chunks:
                multiplexGroupDS = self.create_multiplexed_chunk(waveforms, channel_chunk)
                dataset.WaveformSequence.append(multiplexGroupDS)        

        # each unique frequency will have a different label for the multiplex group. 
        # The multiplex group will 
        # have a separate instance contain the channels with the same frequency
        # each chunk will have channels with the same frequency (with same label).
        # Channels for each multiplex group
        # will have the same start and end time/ sample ids.

        return dataset

    def add_waveform_chunks_by_time(self, dataset, waveforms):

        # dicom waveform -> sequence of multiplex group -> channels wiht same sampling frequency
        # multiplex group can have labels.

        # input here:  each channel can have different sampling rate.  each with chunks
        # group sample frequencies together
        # then into chunks.

        # gather chunks and sort by start time
        chunk_ids = []
        for channel in waveforms.keys():
            chunks = waveforms[channel]['chunks']
            for i in range(len(chunks)):
                chunk_ids.append((chunks[i]['start_time'], channel, i))
        # now sort by start time then channel then chunkid
        chunk_ids.sort(key=lambda x: (x[0], x[1], x[2]))

        print("Chunk Ids", chunk_ids)

        dataset.WaveformSequence = []
        # iterate over frequencies
        for _, channel, chunkid in chunk_ids:

            # iterate over chunks
            multiplexGroupDS = self.create_multiplexed_chunk(waveforms, {channel: [chunkid]})

            dataset.WaveformSequence.append(multiplexGroupDS)

        # each unique frequency will have a different label for the multiplex group.
        # The multiplex group will 
        # have a separate instance contain the channels with the same frequency
        # each chunk will have channels with the same frequency (with same label).
        # Channels for each multiplex group
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
        dicom = self.add_waveform_chunks_multiplexed(dicom, waveforms)        

        # Save DICOM file.  write_like_original is required
        dicom.save_as(path, write_like_original=False)

        # dicom.save_as("/mnt/c/Users/tcp19/Downloads/Compressed/test_waveform.dcm", write_like_original=False)

    def read_waveforms(self, path, start_time, end_time, signal_names):
        """
        have to read the whole data set each time if using dcmread. this is not efficient.
        """
        requested_channels = set(signal_names)

        t1 = time.time()
        dicom = dcmread(path, defer_size = 32)
        t2 = time.time()
        # print("Read time", t2 - t1)

        results = { name: [] for name in signal_names }
        dtype = WAVEFORM_DTYPES[(self.WaveformBitsAllocated, self.WaveformSampleInterpretation)]

        labels = []
        for multiplex_group in dicom.WaveformSequence:
            # check match by channel name, start and end time

            t1 = time.time()
            group_channels = set([channel_def.ChannelSourceSequence[0].CodeMeaning for channel_def in multiplex_group.ChannelDefinitionSequence ])
            if (len(requested_channels.intersection(group_channels)) == 0):
                # print("skipped due to channel:", group_channels, requested_channels)
                continue

            start_t = multiplex_group.MultiplexGroupTimeOffset / 1000.0
            end_t = start_t + float(multiplex_group.NumberOfWaveformSamples) / float(multiplex_group.SamplingFrequency)

            if (start_t >= end_time) or (end_t <= start_time):
                # print("skipped outside range", start_t, end_t, start_time, end_time)
                continue

            # inbound.  compute the time:
            chunk_start_t = max(start_t, start_time)
            chunk_end_t = min(end_t, end_time)

            # # out of bounds.  so exclude.
            # if (chunk_start_t >= chunk_end_t):
            #     print("skipped 0 legnth group ", chunk_start_t, chunk_end_t)
            #     continue

            correction_factors = [channel_def.ChannelSensitivityCorrectionFactor for channel_def in multiplex_group.ChannelDefinitionSequence]

            # now get the data
            nchannels = multiplex_group.NumberOfWaveformChannels
            nsamples = multiplex_group.NumberOfWaveformSamples

            # compute the global and chunk sample offsets.
            if (chunk_start_t == start_t):
                chunk_start_sample = 0
                global_start_sample = np.round(start_t * float(multiplex_group.SamplingFrequency)).astype(int)
            else: 
                chunk_start_sample = np.round((chunk_start_t - start_t) * float(multiplex_group.SamplingFrequency)).astype(int)
                global_start_sample = np.round(chunk_start_t * float(multiplex_group.SamplingFrequency)).astype(int)
            if (chunk_end_t == end_t):
                chunk_end_sample = multiplex_group.NumberOfWaveformSamples
                global_end_sample = global_start_sample + chunk_end_sample
            else:
                chunk_end_sample = np.round((chunk_end_t - start_t) * float(multiplex_group.SamplingFrequency)).astype(int)
                global_end_sample = global_start_sample + chunk_end_sample

            t2 = time.time()
            # print(multiplex_group.MultiplexGroupLabel, chunk_start_sample, chunk_end_sample, "metadata", t2 - t1)

            t1 = time.time()
            raw_arr = np.frombuffer(cast(bytes, multiplex_group.WaveformData), dtype=dtype).reshape([nsamples, nchannels])
            t2 = time.time()
            # print(multiplex_group.MultiplexGroupLabel, chunk_start_sample, chunk_end_sample, "get raw_arr", t2 - t1)

            # print(raw_arr.shape)
            for i, name in enumerate(group_channels):
                if name not in signal_names:
                    continue

                # if name not in results.keys():
                #     # results[name] = {}
                #     # results[name]['chunks'] = []
                #     results[name] = []

                # unit = channel_def.ChannelSensitivityUnitsSequence[0].CodeValue
                # gain = 1.0 / channel_def.ChannelSensitivityCorrectionFactor
                # results[name]['units'] = unit
                # results[name]['samples_per_second'] = multiplex_group.SamplingFrequency
                t1 = time.time()
                mask = (raw_arr[chunk_start_sample:chunk_end_sample, i] == self.PaddingValue)
                arr_i = raw_arr[chunk_start_sample:chunk_end_sample, i].astype(self.MemoryDataType, copy=False) * float(correction_factors[i])             # out of bounds.  so exclude.)
                # arr_i = [ x * float(correction_factors[i]) for x in raw_arr[chunk_start_sample:chunk_end_sample, i] ]
                arr_i[mask] = np.nan
                # arr_i[arr_i <= float(self.PaddingValue) ] = np.nan
                # arr_i = arr_i * float(correction_factors[i])
                t2 = time.time()
                # print("convert ", name, " to float.", t2 - t1, start_t, end_t, start_time, end_time)

                # chunk = {'start_time': chunk_start_t, 'end_time': chunk_end_t, 
                #                              'start_sample': global_start_sample, 'end_sample': global_end_sample,
                #                              'gain': gain, 'samples': arr_i}
                # results[name]['chunks'].append(chunk)
                results[name].append(arr_i)

        for name in results.keys():
            if ( len(results[name]) > 0):
                results[name] = np.concatenate(results[name])

        return results


# dicom value types are constrained by IOD type
# https://dicom.nema.org/medical/dicom/current/output/chtml/part03/PS3.3.html
class DICOMFormat32(BaseDICOMFormat):
    WaveformSampleInterpretation = 'SL'
    WaveformBitsAllocated = 32
    # scalp EEG
    # https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.34.12.4.6.html
    WaveformSampleValueRepresentation = 'SL'
    # electromyogram, electrooculogram, sleep EEG, multichannel repiratory signals, general32bit ECG
    PaddingValue = -2147483648
    FileDatatype = np.int32
    MemoryDataType = np.float32


class DICOMFormat16(BaseDICOMFormat):
    WaveformSampleInterpretation = 'SS'
    WaveformBitsAllocated = 16
    # all waveforms, except basic audio
    # https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.34.3.4.8.html
    WaveformSampleValueRepresentation = 'SS'
    PaddingValue = -32768
    FileDatatype = np.int16
    MemoryDataType = np.float32


class DICOMFormat8(BaseDICOMFormat):
    WaveformSampleInterpretation = 'UB'
    WaveformBitsAllocated = 8
    # fixed body position waveform, basic audio (also MB and AB)
    WaveformSampleValueRepresentation = 'UB'
    PaddingValue = 0
    FileDatatype = np.int8
    MemoryDataType = np.float32


class DICOMFormat8(BaseDICOMFormat):
    WaveformSampleInterpretation = 'SB'
    WaveformBitsAllocated = 8
    # respiratory, gneral audio, realtime audio, ambulatory ECG, arterial pulse
    WaveformSampleValueRepresentation = 'SB'
    PaddingValue = -127
    FileDatatype = np.int8
    MemoryDataType = np.float32
