import os

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom import uid
from pydicom.waveforms.numpy_handler import WAVEFORM_DTYPES
from typing import TYPE_CHECKING, cast

import math
from datetime import datetime

import warnings
# warnings.filterwarnings("error")

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

# the max number of sequence element means that we may not be able to chunk within a file?  12 lead ECG is limited to 15 secs.
# sleep EEG is not limited...
# TODO: 1. the list of channels need to be partitioned to multiple files
#   2. multiple series need to be created, each contains a single modality.
#   3. a series then contains a single chunk potentially (e.g. 12 lead ecg).
#   4. so a longer length of data will be split into multiple files in the same series.
#   5. this means files in a series will need to be opened for random access.



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
# DONE: fix the nans...

# data:  channels grouped in to multiplex groups by standard spec.
#        multiplex groups are grouped into instance files
#        each file map to one segment
#        all files are for one set of waves are grouped into a series
#        each series contains a dicomdir

# TODO: [x] organize output
# TODO: [x] numpy transpose
# TODO: [x] dicomdir write
# TODO: [x] dicomdir read
# TODO: [x] extract with random access
# TODO: [x] merge chunks


# Value Representation Formats. base class
class DICOMWaveformVR:
    ...

# reprensetation by bits
class DICOMWaveform8(DICOMWaveformVR):
    WaveformBitsAllocated = 8
    WaveformSampleInterpretation = "SB"
    PaddingValue = int(-128)
    PythonDatatype = np.int8
               
class DICOMWaveform16(DICOMWaveformVR):
    WaveformBitsAllocated = 16
    WaveformSampleInterpretation = "SS"
    PaddingValue = int(-32768)
    PythonDatatype = np.int16
    
class DICOMWaveform32(DICOMWaveformVR):
    WaveformBitsAllocated = 32
    WaveformSampleInterpretation = "SL"
    PaddingValue = int(-2147483648)
    PythonDatatype = np.int32


# relevant definitions:
# name: modality name, maximum sequences, grouping, SOPUID, max samples, sampling frequency, source, datatype, 
# TwelveLeadECGWaveform:  ECG, 1-5, {1: I,II,III; 2: aVR, aVL, aVF; 3: V1, V2, V3; 4: V4, V5, V6; 5: II}, 16384, 200-1000, DCID 3001 “ECG Lead”, SS
# GeneralECGWaveform: ECG, 1-4, 1-24 per serquence, ?, 200-1000, DCID 3001 “ECG Lead”, SS
# General32BitECGWaveform: ECG, 1-4, 1-24 per serquence, ?, by confirmance statement, DCID 3001 “ECG Lead”, SL
# AmbulatoryECGWaveform:  ECG, 1, 1-12, maxsize of wvaeform data attribute, 50-1000, DCID 3001 “ECG Lead”, SB/SS
# HemodynamicWaveform: HD, 1-4, 1-8, maxsize of wvaeform data attribute, <400, , SS
# CardiacElectrophysiologyWaveform: EPS, 1-4, , <=20000, DCID 3011 “Electrophysiology Anatomic Location” , SS
# ArterialPulseWaveform: HD, 1, 1, ? , <600, DCID 3004 “Arterial Pulse Waveform” , SB/SS
# RespiratoryWaveform: RESP, 1, 1, ? , <100, DCID 3005 “Respiration Waveform”, SB/SS
# ScalpEEGWaveform:  EEG, 1 (interruption as separate instances), 1-64, , unconstrained, DCID 3030 “EEG Lead”, SS/SL
# ElectromyogramWaveform: EMG, unconstrained, 1-64, , unconstrained, DCID 3031 “Lead Location Near or in Muscle” or DCID 3032 “Lead Location Near Peripheral Nerve”, SS/SL
# SleepEEGWaveform: EEG, unconstrained, 1-64, , unconstrained, DCID 3030 “EEG Lead” , SS/SL
# MultichannelRespiratoryWaveform: RESP, >1, , unconstrained, DCID 3005 “Respiration Waveform” , SS/SL
# 

UCUM_ENCODING = {
    "mV": "millivolt",
    "bpm": "beats per minute",
    "mmHg": "millimeter of mercury",
    "uV": "microvolt",
    "NU": "number",
    "Ohm": "ohm"
}

# dicom IODs types
class DICOMWaveformIOD:
    storage_uid = None  
    modality = None
    channel_coding = {} 
    VR = DICOMWaveformVR
 
 
 
    # input is a waveforms dict with only channels that are part of the specified waveform storage object
    # return an array of chunks.  for now, assume all frequencies for a group is the same.
    def group_and_validate(self, waveforms: dict, channels: list) -> list:
        
        grouped_channels = {}
        for ch in channels:
            if ch not in self.channel_coding.keys():
                raise ValueError("channel ", ch, " is not part of the IOD ", type(self))
            
            # get the multiplex group number for the channel
            group = self.channel_coding[ch]['group']
            freq = waveforms[ch]['samples_per_second']
            # organize the channel by group and frequency
            if group not in grouped_channels.keys():
                grouped_channels[group] = { freq: [ch] }
            elif freq not in grouped_channels[group].keys():
                grouped_channels[group][freq] = [ch]
            else:
                grouped_channels[group][freq].append(ch)
        
        print("Grouped channels in group_and_validate ", grouped_channels)
        
        for group, freqs in grouped_channels.items():
            if len(freqs.keys()) > 1:
                print("ERROR:  group ", group, " has multiple freqiuencies: ", freqs.keys())
        
        # in case thre are multiple frequencies for a group of channels that should go together: 
        # if there are more than 1 frequencies in a group, split it out into a separate set.
        # find max distinct frequencies for any group
        max_n_sets = max([len(freqs) for freqs in grouped_channels.values()])
        grouped_channel_sets = [ {} for i in range(max_n_sets)]
        for group, freqs in grouped_channels.items():
            for i, f in enumerate(freqs.items()):
                grouped_channel_sets[i][group] = f
           
        print("Grouped Channel Sets in group_and_validate ", grouped_channel_sets)       
    
        return grouped_channel_sets

    
class TwelveLeadECGWaveform(DICOMWaveformIOD):
    def __init__(self, hifi: bool = False, num_channels: int = None):
        pass
        
    VR = DICOMWaveform16
    storage_uid = uid.TwelveLeadECGWaveformStorage
    modality = 'ECG'
    # https://dicom.nema.org/medical/dicom/current/output/html/part16.html#sect_CID_3001
    channel_coding = {
        "I": {'group' : 1, 'scheme': 'MDC', 'value': '2:1', 'meaning': 'Lead I'},
        "II": {'group' : 1, 'scheme': 'MDC', 'value': '2:2', 'meaning': 'Lead II'},
        "III": {'group' : 1, 'scheme': 'MDC', 'value': '2:61', 'meaning': 'Lead III'},
        "V": {'group' : 3, 'scheme': 'MDC', 'value': '2:3', 'meaning': 'Lead V1'},
        "V1": {'group' : 3, 'scheme': 'MDC', 'value': '2:3', 'meaning': 'Lead V1'},
        "V2": {'group' : 3, 'scheme': 'MDC', 'value': '2:4', 'meaning': 'Lead V2'},
        "V3": {'group' : 3, 'scheme': 'MDC', 'value': '2:5', 'meaning': 'Lead V3'},
        "V4": {'group' : 4, 'scheme': 'MDC', 'value': '2:6', 'meaning': 'Lead V4'},
        "V5": {'group' : 4, 'scheme': 'MDC', 'value': '2:7', 'meaning': 'Lead V5'},
        "V6": {'group' : 4, 'scheme': 'MDC', 'value': '2:8', 'meaning': 'Lead V6'},
        "aVR": {'group' : 2, 'scheme': 'MDC', 'value': '2:62', 'meaning': 'aVR, augmented voltage, right'},
        "aVL": {'group' : 2, 'scheme': 'MDC', 'value': '2:63', 'meaning': 'aVL, augmented voltage, left'},
        "aVF": {'group' : 2, 'scheme': 'MDC', 'value': '2:64', 'meaning': 'aVF, augmented voltage, foot'},
    }
    
    
class GeneralECGWaveform(DICOMWaveformIOD):
    def __init__(self, hifi: bool = False, num_channels: int = None):
        if hifi:
            self.VR = DICOMWaveform32
            self.storage_uid = '1.2.840.10008.5.1.4.1.1.9.1.4'
        else:
            self.VR = DICOMWaveform16
            self.storage_uid = uid.GeneralECGWaveformStorage
        
    modality = 'ECG'
    channel_coding = {
        "I": {'group' : 1, 'scheme': 'MDC', 'value': '2:1', 'meaning': 'Lead I'},
        "II": {'group' : 1, 'scheme': 'MDC', 'value': '2:2', 'meaning': 'Lead II'},
        "III": {'group' : 1, 'scheme': 'MDC', 'value': '2:61', 'meaning': 'Lead III'},
        "V": {'group' : 3, 'scheme': 'MDC', 'value': '2:3', 'meaning': 'Lead V1'},
        "V1": {'group' : 3, 'scheme': 'MDC', 'value': '2:3', 'meaning': 'Lead V1'},
        "V2": {'group' : 3, 'scheme': 'MDC', 'value': '2:4', 'meaning': 'Lead V2'},
        "V3": {'group' : 3, 'scheme': 'MDC', 'value': '2:5', 'meaning': 'Lead V3'},
        "V4": {'group' : 4, 'scheme': 'MDC', 'value': '2:6', 'meaning': 'Lead V4'},
        "V5": {'group' : 4, 'scheme': 'MDC', 'value': '2:7', 'meaning': 'Lead V5'},
        "V6": {'group' : 4, 'scheme': 'MDC', 'value': '2:8', 'meaning': 'Lead V6'},
        "aVR": {'group' : 2, 'scheme': 'MDC', 'value': '2:62', 'meaning': 'aVR, augmented voltage, right'},
        "aVL": {'group' : 2, 'scheme': 'MDC', 'value': '2:63', 'meaning': 'aVL, augmented voltage, left'},
        "aVF": {'group' : 2, 'scheme': 'MDC', 'value': '2:64', 'meaning': 'aVF, augmented voltage, foot'},
    }
    

class AmbulatoryECGWavaform(DICOMWaveformIOD):
    def __init__(self, hifi: bool = False, num_channels: int = None):
        pass
        
    # 8bit allowed, but gain may be too high for 8 bit
    VR = DICOMWaveform16
    storage_uid = uid.AmbulatoryECGWaveformStorage
    modality = 'ECG'
    channel_coding = {
        'ECG': {'group': 1},
    }

class CardiacElectrophysiologyWaveform(DICOMWaveformIOD):
    def __init__(self, hifi: bool = False, num_channels: int = None):
        pass
    
    VR = DICOMWaveform16
    storage_uid = uid.CardiacElectrophysiologyWaveformStorage
    modality = 'EPS'
    channel_coding = {
        'EPS': {'group': 1},
    }

class HemodynamicWavaform(DICOMWaveformIOD):
    def __init__(self, hifi: bool = False, num_channels: int = None):
        pass
    
    VR = DICOMWaveform16
    storage_uid = uid.HemodynamicWaveformStorage
    modality = 'HD'
    channel_coding = {
        'HD': {'group': 1},
    }

    
class ArterialPulseWaveform(DICOMWaveformIOD):
    def __init__(self, hifi: bool = False, num_channels: int = None):
        # if hifi:
        #     self.VR = DICOMWaveform16
        # else:
        #     self.VR = DICOMWaveform8
        pass
                
    # 8bit allowed, but gain may be too high for 8 bit
    VR = DICOMWaveform16                
    storage_uid = uid.ArterialPulseWaveformStorage
    modality = 'HD'
    channel_coding = {
        # to fix.
        'Pleth': {'group': 1, 'scheme': 'SCPECG', 'value': '5.6.3-9-00', 'meaning': 'Plethysmogram'},
        'PLETH': {'group': 1, 'scheme': 'SCPECG', 'value': '5.6.3-9-00', 'meaning': 'Plethysmogram'}
    }

    
class RespiratoryWaveform(DICOMWaveformIOD):
    def __init__(self, hifi: bool = False, num_channels: int = None):
        if num_channels <= 1:
            # if hifi:
            #     self.VR = DICOMWaveform16
            # else:
            #     self.VR = DICOMWaveform8
            # 8bit allowed, but gain may be too high for 8 bit
            self.VR = DICOMWaveform16
            storage_uid = uid.RespiratoryWaveformStorage
        elif num_channels > 1:
            if hifi:
                self.VR = DICOMWaveform32
            else:
                self.VR = DICOMWaveform16
            storage_uid = uid.MultichannelRespiratoryWaveformStorage
                    
    modality = 'RESP'
    channel_coding = {
        'RESP': {'group': 1, 'scheme': 'SCPECG', 'value': '5.6.3-9-01', 'meaning': 'Respiration'},
        'Resp': {'group': 1, 'scheme': 'SCPECG', 'value': '5.6.3-9-01', 'meaning': 'Respiration'}
    }
        
class RoutineScalpEEGWaveform(DICOMWaveformIOD):
    def __init__(self, hifi: bool = False, num_channels: int = None):
        if hifi:
            self.VR = DICOMWaveform32
        else:
            self.VR = DICOMWaveform16
        
    storage_uid = uid.RoutineScalpElectroencephalogramWaveformStorage
    modality = 'EEG'
    channel_coding = {
        'EEG': {'group': 1},
    }

class SleepEEGWaveform(DICOMWaveformIOD):
    def __init__(self, hifi: bool = False, num_channels: int = None):
        if hifi:
            self.VR = DICOMWaveform32
        else:
            self.VR = DICOMWaveform16
        
    storage_uid = uid.SleepElectroencephalogramWaveformStorage
    modality = 'EEG'
    channel_coding = {
        'EEG': {'group': 1},
    }
    
class ElectromyogramWaveform(DICOMWaveformIOD):
    def __init__(self, hifi: bool = False, num_channels: int = None):
        if hifi:
            self.VR = DICOMWaveform32
        else:
            self.VR = DICOMWaveform16
        
    storage_uid = uid.ElectromyogramWaveformStorage
    modality = 'EMG'
    channel_coding = {
        'EMG': {'group': 1},
    }



class DICOMWaveformWriter:
    
    # Currently, this class uses a single segment and stores many
    # signals in one signal file.  Using multiple segments and
    # multiple signal files could improve efficiency of storage and
    # per-channel access.
    
        
    # create the outer container corresponding to the target waveform
    # note that each series containes files of the same modality.
    def make_empty_wave_filedataset(self, 
                                    waveformType: DICOMWaveformIOD):
        
        # file metadata for fast access.
        fileMeta = FileMetaDataset()
        fileMeta.MediaStorageSOPClassUID = waveformType.storage_uid
        fileMeta.MediaStorageSOPInstanceUID = uid.generate_uid()
        fileMeta.TransferSyntaxUID = uid.ExplicitVRLittleEndian   # this is hardcoded.

        fileDS = Dataset()
        fileDS.file_meta = fileMeta
        
        # Mandatory.  can we reuse SOP INstance Uid?
        fileDS.SOPInstanceUID = fileMeta.MediaStorageSOPInstanceUID
        ## this should be same as the MediaStorageSOPClassUID
        fileDS.SOPClassUID = fileMeta.MediaStorageSOPClassUID
        fileDS.is_little_endian = True
        fileDS.is_implicit_VR = True
                
        #TODO: should we put the list of channels here? and start and end times?

        return fileDS


    def set_order_info(self, dataset, 
                       referringPhysicianName = "Physician^Jane", 
                       accessionNumber = "0000"):
        # General Study Module
        dataset.ReferringPhysicianName = referringPhysicianName
        dataset.AccessionNumber = accessionNumber
        return dataset

    def set_study_info(self, dataset, 
                        patientID = "N/A", 
                        patientName = "Doe^Jill",
                        patientBirthDate = "19800101",
                        patientSex = "F",
                        studyID = 0,
                        studyUID = None,
                        studyDate: datetime = None):
        # patient module:
        dataset.PatientID = patientID
        dataset.PatientName = patientName
        dataset.PatientBirthDate = patientBirthDate
        dataset.PatientSex = patientSex
        
        # needed to build DICOMDIR
        t = datetime.now() if studyDate is None else studyDate
        dataset.StudyDate = t.strftime('%Y%m%d')
        dataset.StudyTime = t.strftime('%H%M%S')
        
        # General Study Module
        dataset.StudyInstanceUID = uid.generate_uid() if studyUID is None else studyUID
        dataset.StudyID = str(studyID)
                
        return dataset
    
    # seriesNumber should increment.   seriesUID should be generated.
    def set_series_info(self, dataset, 
                        waveformType: DICOMWaveformIOD,
                        seriesUID = None,
                        seriesNumber = 0):
        dataset.Modality = waveformType.modality
        dataset.SeriesNumber = seriesNumber
        # General Study Module
        dataset.SeriesInstanceUID = uid.generate_uid() if seriesUID is None else seriesUID
        
        return dataset
    
    def set_waveform_acquisition_info(self, dataset, 
                                      manufacturer = "Unknown",
                                      modelName = "Unknown",
                                      acqDate: datetime = None,
                                      instanceNumber = 1):
        # General Equipment Module
        dataset.Manufacturer = manufacturer
        
        # Synchronization Module
        dataset.AcquisitionTimeSynchronized = 'Y'
        dataset.SynchronizationTrigger = "NO TRIGGER"
        dataset.SynchronizationFrameOfReferenceUID = "1.2.840.10008.15.1.1"  #UTC. https://dicom.innolitics.com/ciods/general-ecg/synchronization/00200200
        
        # Waveform Identification Module
        dataset.InstanceNumber = instanceNumber
        t = datetime.now() if acqDate is None else acqDate
        dataset.ContentDate = t.strftime('%Y%m%d')
        dataset.ContentTime = t.strftime('%H%M%S')
        dataset.AcquisitionDateTime = t.strftime('%Y%m%d%H%M%S')
        
        # Acquisition Context Module
        # TODO:  these need to be modified.
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
    
    
    # channel_chunks is a list of tuples (channel, chunk).
    # 
    def create_multiplexed_chunk(self, waveforms: dict, 
                                 iod: DICOMWaveformIOD, 
                                 group: int,
                                 channel_chunk: list,
                                 start_time: float,
                                 end_time: float):
                
        # verify that all chunks are in bound
        if any([ chunk >= len(waveforms[channel]['chunks']) for (channel, chunk) in channel_chunk ]):
            raise ValueError("Channel chunk out of bounds:" + channel + " " + chunk)
        
        multiplex_chunk_freqs = { channel: waveforms[channel]['samples_per_second'] for (channel, _) in channel_chunk }
        
        # test that the frequencies are all the same
        if len(set(multiplex_chunk_freqs.values())) != 1:
            print("ERROR: Chunks with different freqs", multiplex_chunk_freqs)
            raise ValueError("Chunks have different frequencies")
        
        freq = next(iter(multiplex_chunk_freqs.values()))
        
        out_start = int(np.round(start_time * freq))
        out_end = int(np.round(end_time * freq))
        
        chunk_max_len = out_end - out_start
        
        unprocessed_chunks = []
        channels = {}
        i = 0
        for (channel, chunkid) in channel_chunk:
            # assign ids for each channel
            if channel not in channels.keys():
                channels[channel] = i
                i += 1
            
            # use samples - the sample ids are about same as time * freq
            c_start = waveforms[channel]['chunks'][chunkid]['start_sample']
            c_end = waveforms[channel]['chunks'][chunkid]['end_sample']
            
            st = max(c_start, out_start)
            et = min(c_end, out_end)
            
            channel_start = 0 if (st == c_start) else (st - c_start)
            target_start = 0 if (st == out_start) else (st - out_start)
            
            duration = et - st
                        
            unprocessed_chunks.append((channel, chunkid, channel_start, target_start, target_start + duration))
            
        # create a new multiplex group
        
        # waveform module.  see https://dicom.innolitics.com/ciods/general-ecg/waveform/54000100/003a0200/003a0319
        # shared for multiple waveforms.
        
        # endtime is implicit: start_time + number of samples / sampling frequency
        
        wfDS = Dataset()  # this is a multiplex group...
        # in milliseconds.  can only be 16 digits long.  1 year has 31565000000 milliseconds.  we can leave 4 decimal places. and still be within 16 digitas and can handle 3 years.
        wfDS.MultiplexGroupTimeOffset = str(np.round(1000.0 * start_time, decimals=4) )  # first start point
        # wfDS.TriggerTimeOffset = '0.0'
        wfDS.WaveformOriginality = "ORIGINAL"
        wfDS.NumberOfWaveformChannels = len(channels.keys())
        wfDS.SamplingFrequency = freq
        wfDS.NumberOfWaveformSamples = int(chunk_max_len)
        wfDS.MultiplexGroupLabel = str(group)
        wfDS.WaveformBitsAllocated = iod.VR.WaveformBitsAllocated
        wfDS.WaveformSampleInterpretation = iod.VR.WaveformSampleInterpretation
        wfDS.WaveformPaddingValue = iod.VR.PaddingValue.to_bytes(4, 'little', signed=True)  # OW type.  4 bytes.
        
        # channel definitions
        # channel def and samples should be in the same order.
        channeldefs = {}
        samples = np.full(shape = (len(channels.keys()), chunk_max_len), 
                          fill_value = iod.VR.PaddingValue,
                          dtype=iod.VR.PythonDatatype)
        # now collect the chunks into an array, generate the multiplex group, and increment the chunk ids.
        unprocessed_chunks.sort(key=lambda x: (x[0], x[3], x[4])) # ordered by channel
        for channel, chunk_id, start_src, start_target, end_target in unprocessed_chunks: 
            
            chunk = waveforms[channel]['chunks'][chunk_id]
            duration = end_target - start_target
            end_src = start_src + duration
            
            # print("channel start ", start_src, end_src, start_target, end_target, duration)
            
            # QUANTIZE: and pad missing data
            # values is in original type.  nan replaced with PaddingValue then will be multiplied by gain, so divide here to avoid overflow.
            # values = np.nan_to_num(np.frombuffer(chunk['samples'][start_src:end_src], dtype=np.dtype(chunk['samples'].dtype)), 
            #                        nan = float(iod.VR.PaddingValue) / float(chunk['gain']) )
            v = np.frombuffer(chunk['samples'][start_src:end_src], dtype=np.dtype(chunk['samples'].dtype))
            gain = float(chunk['gain'])
            values = np.where(np.isnan(v), float(iod.VR.PaddingValue), v * gain)


            chan_id = channels[channel]
            # write out in integer format
            # samples[chan_id][start_target:end_target] = np.round(values * float(chunk['gain']), decimals=0).astype(iod.VR.PythonDatatype)
            samples[chan_id][start_target:end_target] = np.round(values, decimals=0).astype(iod.VR.PythonDatatype)
            # print("chunk shape:", chunk['samples'].shape)
            # print("values shape: ", values.shape)
            # print("samples shape: ", samples.shape)
            
        # interleave the samples
        samplesT = np.transpose(samples)
        # print(channel_chunk)
        # print("output shape", samplesT.shape)
        
        
        for (channel, chunk_id) in channel_chunk:
            if channel in channeldefs.keys():
                continue
                
            chunk = waveforms[channel]['chunks'][chunk_id]        
            unit = waveforms[channel]['units']
                
            # create the channel
            chdef = Dataset()
            # chdef.ChannelTimeSkew = '0'  # Time Skew OR Sample Skew
            chdef.ChannelSampleSkew = "0"
            # chdef.ChannelOffset = '0'
            chdef.WaveformBitsStored = iod.VR.WaveformBitsAllocated
            chdef.ChannelSourceSequence = [Dataset()]
            source = chdef.ChannelSourceSequence[0]
                
            
            # this needs a look up from a controlled vocab.  This is not correct here..
            source.CodeValue = iod.channel_coding[channel]['value']
            source.CodingSchemeDesignator = iod.channel_coding[channel]['scheme']
            source.CodingSchemeVersion = "unknown"
            source.CodeMeaning = channel
                
            chdef.ChannelSensitivity = 1.0
            chdef.ChannelSensitivityUnitsSequence = [Dataset()]
            units = chdef.ChannelSensitivityUnitsSequence[0]
                
            # this also needs a look up from a controlled vocab.  This is not correct here...
            units.CodeValue = unit
            units.CodingSchemeDesignator = "UCUM"
            units.CodingSchemeVersion = "1.4"
            units.CodeMeaning = UCUM_ENCODING[unit]  # this needs to be fixed.
                
            # multiplier to apply to the encoded value to get back the orginal input.
            ds = str(float(1.0) / float(chunk['gain']))
            chdef.ChannelSensitivityCorrectionFactor = ds if len(ds) <= 16 else ds[:16]
            chdef.ChannelBaseline = '0'
            chdef.WaveformBitsStored = iod.VR.WaveformBitsAllocated
            # only for amplifier type of AC
            # chdef.FilterLowFrequency = '0.05'
            # chdef.FilterHighFrequency = '300'
            channeldefs[channel] = chdef            
            

        wfDS.ChannelDefinitionSequence = channeldefs.values() 
        # actual bytes. arr is a numpy array of shape np.stack((ch1, ch2,...), axis=1)
        # arr = np.stack([ samples[channel] for channel in channel_chunk.keys()], axis=1)
        wfDS.WaveformData = samplesT.tobytes()

        return wfDS

    
    def add_waveform_chunks_multiplexed(self, dataset,
                                        iod: DICOMWaveformIOD,
                                        chunk_info: dict,
                                        waveforms: dict):
        
        # dicom waveform -> sequence of multiplex group -> channels with same sampling frequency
        # multiplex group can have labels.
        # {start_t, end_t, {(channel, chunk_id) : group, ...}}
        
        # process the channel chunks
        dataset.WaveformSequence = []
        start_time = chunk_info['start_t']
        end_time = chunk_info['end_t']
        channel_chunks = chunk_info['channel_chunk']
            
        # group by channel groups
        unique_groups = set(channel_chunks.values())
        grouped_channels = { group: [ key for key, val in channel_chunks.items() if val == group ] for group in unique_groups }
        
        for group, chanchunks in grouped_channels.items():
            # input to this is [(channel, chunk), ... ]
            multiplexGroupDS = self.create_multiplexed_chunk(waveforms, iod, group, chanchunks, 
                                                            start_time=start_time, end_time=end_time)
            dataset.WaveformSequence.append(multiplexGroupDS)        
        
        # each unique frequency will have a different label for the multiplex group..  The multiplex group will 
        # have a separate instance contain the channels with the same frequency
        # each chunk will have channels with the same frequency (with same label).  Channels for each multiplex group
        # will have the same start and end time/ sample ids.
        
        return dataset

    

# dicom value types are constrained by IOD type
# https://dicom.nema.org/medical/dicom/current/output/chtml/part03/PS3.3.html

