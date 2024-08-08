import os

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom import uid
from pydicom.waveforms.numpy_handler import WAVEFORM_DTYPES
from typing import TYPE_CHECKING, cast

import math
from datetime import datetime

import warnings

from waveform_benchmark.formats.base import BaseFormat

# dicom3tools currently does NOT validate the IODs for waveform.  IT does validate the referencedSOPClassUIDInFile in DICOMDIR file.

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

# TODO: [x] may need to separate channels that are not ECGs in other files.
# TODO: [x] private tags for speeding up metadata access while still maintain compatibility?  this would provide "what do we have", and not necessarily "where is it in the file"
# TODO: compression - in transfer syntax?  likely not standards compliant.
# TODO: [x] random access support?
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

# NOTE: float32 IEEE standard represents values like -2147483600.0 as -2147483648.0.
#       not an issue for 8bit and 16bit output, for 32bit, use double (64bit) that works.

# Value Representation Formats. base class
class DICOMWaveformVR:
    ...

# reprensetation by bits
class DICOMWaveform8(DICOMWaveformVR):
    WaveformBitsAllocated = 8
    WaveformSampleInterpretation = "SB"
    FileDatatype = np.int8
    FloatDataType = np.float32
    PaddingValue = np.iinfo(FileDatatype).min
               
class DICOMWaveform16(DICOMWaveformVR):
    WaveformBitsAllocated = 16
    WaveformSampleInterpretation = "SS"
    FileDatatype = np.int16
    FloatDataType = np.float32
    PaddingValue = np.iinfo(FileDatatype).min
    
class DICOMWaveform32(DICOMWaveformVR):
    WaveformBitsAllocated = 32
    WaveformSampleInterpretation = "SL"
    FileDatatype = np.int32
    FloatDataType = np.float64
    PaddingValue = np.iinfo(FileDatatype).min


# relevant definitions:
# name: modality name, maximum sequences, grouping, SOPUID, max samples, sampling frequency, source, datatype, 
# TwelveLeadECGWaveform:  ECG, 1-5, {1: I,II,III; 2: aVR, aVL, aVF; 3: V1, V2, V3; 4: V4, V5, V6; 5: II}, 16384, 200-1000, DCID 3001 “ECG Lead”, SS
# GeneralECGWaveform: ECG, 1-4, 1-24 per sequence, ?, 200-1000, DCID 3001 “ECG Lead”, SS
# General32BitECGWaveform: ECG, 1-4, 1-24 per sequence, ?, by conformance statement, DCID 3001 “ECG Lead”, SL
# AmbulatoryECGWaveform:  ECG, 1, 1-12, maxsize of waveform data attribute, 50-1000, DCID 3001 “ECG Lead”, SB/SS
# HemodynamicWaveform: HD, 1-4, 1-8, maxsize of waveform data attribute, <400, , SS
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
    "NU": "nadir upstroke",
    "Ohm": "ohm",
    "%": "percent",
}

# dicom IODs types
class DICOMWaveformIOD:
    storage_uid = None  
    modality = None
    channel_coding = {} 
    VR = DICOMWaveformVR

    
class TwelveLeadECGWaveform(DICOMWaveformIOD):
    def __init__(self, bits: int = 16, num_channels: int = None):
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
        "AVR": {'group' : 2, 'scheme': 'MDC', 'value': '2:62', 'meaning': 'aVR, augmented voltage, right'},
        "AVL": {'group' : 2, 'scheme': 'MDC', 'value': '2:63', 'meaning': 'aVL, augmented voltage, left'},
        "AVF": {'group' : 2, 'scheme': 'MDC', 'value': '2:64', 'meaning': 'aVF, augmented voltage, foot'},
    }
    
# 4 groups, 1 to 24 channels each. unknown sample count limit., f in 200-1000
class GeneralECGWaveform(DICOMWaveformIOD):
    def __init__(self, bits: int = 16, num_channels: int = None):
        if bits > 16:
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
        "AVR": {'group' : 2, 'scheme': 'MDC', 'value': '2:62', 'meaning': 'aVR, augmented voltage, right'},
        "AVL": {'group' : 2, 'scheme': 'MDC', 'value': '2:63', 'meaning': 'aVL, augmented voltage, left'},
        "AVF": {'group' : 2, 'scheme': 'MDC', 'value': '2:64', 'meaning': 'aVF, augmented voltage, foot'},
        "MCL": {'group' : 2, 'scheme': 'unknown', 'value': '0', 'meaning': 'MCL, mock circulatory loop'},
    }
    

class AmbulatoryECGWaveform(DICOMWaveformIOD):
    def __init__(self, bits : int = 16, num_channels: int = None):
        pass
        
    # 8bit allowed, but gain may be too high for 8 bit
    VR = DICOMWaveform16
    storage_uid = uid.AmbulatoryECGWaveformStorage
    modality = 'ECG'
    channel_coding = {
        "ECG": {'group' : 4, 'scheme': 'unknown', 'value': '0', 'meaning': 'ECG, Generic ECG lead'},
    }

class CardiacElectrophysiologyWaveform(DICOMWaveformIOD):
    def __init__(self, bits : int = 16, num_channels: int = None):
        pass
    
    VR = DICOMWaveform16
    storage_uid = uid.CardiacElectrophysiologyWaveformStorage
    modality = 'EPS'
    channel_coding = {
        'EPS': {'group': 1},
    }

# max 4 sequences, up to 8 channels each, num of samples limited by waveform maxsize. f < 400,
class HemodynamicWaveform(DICOMWaveformIOD):
    def __init__(self, bits : int = 16, num_channels: int = None):
        pass
    
    VR = DICOMWaveform16
    storage_uid = uid.HemodynamicWaveformStorage
    modality = 'HD'
    channel_coding = {
        "CVP": {'group' : 1, 'scheme': 'unknown', 'value': '0', 'meaning': 'Central Venous Pressure'},
        "CVP1": {'group' : 1, 'scheme': 'unknown', 'value': '0', 'meaning': 'Central Venous Pressure'},
        "CVP2": {'group' : 1, 'scheme': 'unknown', 'value': '0', 'meaning': 'Central Venous Pressure'},
        "PAP": {'group' : 2, 'scheme': 'unknown', 'value': '0', 'meaning': 'Pulmonary Arterial Pressure'},
        "PA2": {'group' : 2, 'scheme': 'unknown', 'value': '0', 'meaning': 'Pulmonary Arterial Pressure'},
        "ABP": {'group' : 3, 'scheme': 'unknown', 'value': '0', 'meaning': 'Ambulatory Blood Pressure'},
        "AO":  {'group' : 4, 'scheme': 'unknown', 'value': '0', 'meaning': 'Aortic Pressure'},
        "ICP": {'group' : 2, 'scheme': 'unknown', 'value': '0', 'meaning': 'Intracranial Pressure'},
        "AR1": {'group' : 2, 'scheme': 'unknown', 'value': '0', 'meaning': 'Aortic Regurgitation Pressure'},
        "AR2": {'group' : 2, 'scheme': 'unknown', 'value': '0', 'meaning': 'Aortic Regurgitation Pressure'},
    }

    
# max 1 sequence, 1 wave each. unknown sample count limit. f < 600.
class ArterialPulseWaveform(DICOMWaveformIOD):
    def __init__(self, bits : int = 16, num_channels: int = None):
        # if bits > 8:
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
        'PLETH': {'group': 1, 'scheme': 'SCPECG', 'value': '5.6.3-9-00', 'meaning': 'Plethysmogram'},
        'SAO2': {'group': 1, 'scheme': 'unknown', 'value': '0', 'meaning': 'Arterial O2 Saturation'},
        'SPO2': {'group': 1, 'scheme': 'unknown', 'value': '0', 'meaning': 'Peripheral Arterial O2 Saturation'}
    }

    
# different IOD for multiple channels.  1 multplex group, 1 channel each. unknown sample count limit. f < 100.
class RespiratoryWaveform(DICOMWaveformIOD):
    def __init__(self, bits : int = 16, num_channels: int = None):
        if num_channels <= 1:
            # if bits > 8:
            #     self.VR = DICOMWaveform16
            # else:
            #     self.VR = DICOMWaveform8
            # 8bit allowed, but gain may be too high for 8 bit
            self.VR = DICOMWaveform16
            self.storage_uid = uid.RespiratoryWaveformStorage
        elif num_channels > 1:
            if bits > 16:
                self.VR = DICOMWaveform32
            else:
                self.VR = DICOMWaveform16
            self.storage_uid = uid.MultichannelRespiratoryWaveformStorage
                                        
    modality = 'RESP'
    channel_coding = {
        'RESP': {'group': 1, 'scheme': 'SCPECG', 'value': '5.6.3-9-01', 'meaning': 'Respiration'},
        'RR': {'group': 1, 'scheme': 'SCPECG', 'value': '5.6.3-9-01', 'meaning': 'Respiration'},
        'ABD' : {'group': 2, 'scheme': 'unknown', 'value': '0', 'meaning': 'Respiration'},
        'CHEST' : {'group': 3, 'scheme': 'unknown', 'value': '1', 'meaning': 'Respiration'},
        'AIRFLOW' : {'group': 4, 'scheme': 'unknown', 'value': '2', 'meaning': 'Respiration'},        
        'CO2' : {'group': 4, 'scheme': 'unknown', 'value': '3', 'meaning': 'CO2 Concentration/Partial Pressure'},
    }
        
class RoutineScalpEEGWaveform(DICOMWaveformIOD):
    def __init__(self, bits : int = 16, num_channels: int = None):
        if bits > 16:
            self.VR = DICOMWaveform32
        else:
            self.VR = DICOMWaveform16
        
    storage_uid = uid.RoutineScalpElectroencephalogramWaveformStorage
    modality = 'EEG'
    channel_coding = {
        'EEG': {'group': 1},
    }

# unlimited number of multiplex groups, up to 64 channels each.  sample size and f unconstrained.
class SleepEEGWaveform(DICOMWaveformIOD):
    def __init__(self, bits : int = 16, num_channels: int = None):
        if bits > 16:
            self.VR = DICOMWaveform32
        else:
            self.VR = DICOMWaveform16
        
    storage_uid = uid.SleepElectroencephalogramWaveformStorage
    modality = 'EEG'
    channel_coding = {
        "F3-M2": {'group': 1, 'scheme': 'unknown', 'value': '1', 'meaning': 'F3-M2 EEG lead'},
        "F4-M1": {'group': 1, 'scheme': 'unknown', 'value': '2', 'meaning': 'F4-M1 EEG lead'},
        "C3-M2": {'group': 1, 'scheme': 'unknown', 'value': '3', 'meaning': 'C3-M2 EEG lead'},
        "C4-M1": {'group': 1, 'scheme': 'unknown', 'value': '4', 'meaning': 'C4-M1 EEG lead'},
        "O1-M2": {'group': 1, 'scheme': 'unknown', 'value': '5', 'meaning': 'O1-M2 EEG lead'},
        "O2-M1": {'group': 1, 'scheme': 'unknown', 'value': '6', 'meaning': 'O2-M1 EEG lead'},
        "E1-M2": {'group': 1, 'scheme': 'unknown', 'value': '7', 'meaning': 'E1-M2 EEG lead'},
    }
    
#unlimited multiplex groups, up to 64 channels each.  sample size and f unconstrained.
class ElectromyogramWaveform(DICOMWaveformIOD):
    def __init__(self, bits : int = 16, num_channels: int = None):
        if bits > 16:
            self.VR = DICOMWaveform32
        else:
            self.VR = DICOMWaveform16
        
    storage_uid = uid.ElectromyogramWaveformStorage
    modality = 'EMG'
    channel_coding = {
        'CHIN1-CHIN2': {'group': 1, 'scheme': 'unknown', 'value': '1', 'meaning': '??'},
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
        mins = {}
        maxs = {}
        gains = {}
        i = 0
        for (channel, chunkid) in channel_chunk:
            
            # use samples - the sample ids are about same as time * freq
            c_start = waveforms[channel]['chunks'][chunkid]['start_sample']
            c_end = waveforms[channel]['chunks'][chunkid]['end_sample']
            
            st = max(c_start, out_start)
            et = min(c_end, out_end)
            
            channel_start = 0 if (st == c_start) else (st - c_start)
            target_start = 0 if (st == out_start) else (st - out_start)
            
            duration = et - st
                        
            # only process if there is finite data.        
            temp_data = waveforms[channel]['chunks'][chunkid]['samples'][channel_start:channel_start+duration]
            if (temp_data is None) or (len(temp_data) == 0):
                continue
            # in case it's all nans or infs
            all_nan = (not np.any(np.isfinite(temp_data)))

            # assign ids for each channel
            if channel not in channels.keys():
                channels[channel] = i
                mins[channel] = []
                maxs[channel] = []
                gains[channel] = []
                i += 1

            mins[channel].append(0 if all_nan else np.fmin.reduce(temp_data))
            maxs[channel].append(0 if all_nan else np.fmax.reduce(temp_data))
            gains[channel].append(waveforms[channel]['chunks'][chunkid]['gain'])
            unprocessed_chunks.append((channel, chunkid, channel_start, target_start, target_start + duration))

        if len(channels) == 0:
            return None

        for c in channels.keys():
            mins[c] = np.fmin.reduce(mins[c])
            maxs[c] = np.fmax.reduce(maxs[c])
            gains[c] = list(set(gains[c]))
            if len(gains[c]) > 1:
                print("ERROR Channel " + c + " has multiple gains ")
                for g in gains[c]:
                    print("gain: " + str(g))    
                raise ValueError("Channel has multiple gains ")
            gains[c] = gains[c][0]
            
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
                          dtype=iod.VR.FileDatatype)
        # now collect the chunks into an array, generate the multiplex group, and increment the chunk ids.
        unprocessed_chunks.sort(key=lambda x: (x[0], x[3], x[4])) # ordered by channel
        
        float_type = iod.VR.FloatDataType

        for channel, chunk_id, start_src, start_target, end_target in unprocessed_chunks: 
            
            chunk = waveforms[channel]['chunks'][chunk_id]
            duration = end_target - start_target
            end_src = start_src + duration
            
            # print("channel start ", start_src, end_src, start_target, end_target, duration)
            
            # QUANTIZE: and pad missing data
            
            ## input type
            #   input is same as "nominal" in dicom parlance.
            #   baseline = gain * (max + min) / 2
            #   digital values are the binary stored data.  digital values = input * gain - baseline. this appears to be by convention.
            #   
            ## per dicom standard.   
            #   channelSensitivity is defined to _include_ both gain and adc resolution.  but probably not "gain * adc resolution"
            #   by standards definition, stored value * channelsensitivity = nominal value in unit specified (e.g. mV).  
            #   this means sensitivity = 1/gain.
            #   further: nominal value * sensitivity correction factor = calibrated value.
            #   baseline:  offset of sample 0 vs actual 0, in nominal value unit. 
            #
            ## harmonizing: 
            #   nominal == input
            #   baseline = -(max + min)/2
            #   sensitivity == gain
            #   stored value (digital) = nominal * gain 
            #   but may not have enough dynamic range in the stored values
            #      rounding error of 0.5 can propagate with a low gain.
            # scale further so we reach the dynamic range.
            
            # get the input values
            v = np.frombuffer(chunk['samples'][start_src:end_src], dtype=np.dtype(chunk['samples'].dtype)).astype(float_type)
            
            # vmin, vmax, gain = minmax[channel]
            vmin = float_type(mins[channel])
            vmax = float_type(maxs[channel])
            gain = float_type(gains[channel])
            
            # print(str(vmin) + "," + str(vmax) + "," + str(gain))
            baseline = (vmin + vmax) * float_type(-0.5)
            
            vg_min = (vmin + baseline ) * gain
            vg_max = (vmax + baseline ) * gain
            vg_mag = np.fmax.reduce([np.abs(vg_min), np.abs(vg_max)])
            # need to leave some room for rounding
            dt_mag = float_type(np.fmin.reduce([np.abs(iod.VR.PaddingValue + 1), np.abs(np.iinfo(iod.VR.FileDatatype).max)]) - 1)
            
            # if np.isnan(vmin) or np.isnan(vmax):
            #     print(v)
            #     print("min ", vmin, " max ", vmax, " size ", len(v), " num of nans ", np.sum(np.isnan(v)))
            #     print(np.sum(np.isnan(chunk['samples'][start_src:end_src])))
            #     print(channel_chunk)
            #     print("start ", start_src, " end ", end_src)
            
            scale = float_type(1.0)
            if (vg_mag != 0) and (dt_mag != 0):
                scale *= (dt_mag / vg_mag)
            
            chan_id = channels[channel]
            y = np.round((v + baseline) * gain * scale, decimals=0)
            # print("min and max " + str(np.fmin.reduce(y)) + "," + str(np.fmax.reduce(y)) + " val max = " + str(float_type(np.iinfo(iod.VR.FileDatatype).max)) + " val min = " + str(float_type(iod.VR.PaddingValue)))
            # if np.any(y > float_type(np.iinfo(iod.VR.FileDatatype).max)):
            #     print("WARNING of max " + channel + "," + str(chunk_id) + " max " + str(float_type(np.iinfo(iod.VR.FileDatatype).max)) + " actual " + str(y[y >= float_type(np.iinfo(iod.VR.FileDatatype).max)][0]))
            # if np.any(y < float_type(iod.VR.PaddingValue)):
            #     print("WARNING of < min " + channel + "," + str(chunk_id) + " min " + str(float_type(np.iinfo(iod.VR.FileDatatype).min)) + " actual " + str(y[y < iod.VR.PaddingValue][0]))
            # if np.any(y == float_type(iod.VR.PaddingValue)):
            #     min_vals = list(set(y[np.where(y == iod.VR.PaddingValue)]))
            #     min_val = min_vals[0]
            #     print("WARNING chann " + channel + " of == min " + channel + "," + str(chunk_id) + " min " + str(float_type(np.iinfo(iod.VR.FileDatatype).min)) + " actual " + str(min_val) + " actual cast to float " + str(np.float32(min_val)))
            #     print("    ," + str(min_val*1.0) + "," + str(float_type(min_val)) + "," + str(min_val) + ", " + str(np.float32(min_val)))
            
            x = np.where(np.isnan(y), float_type(iod.VR.PaddingValue), y)
            # if np.any(np.isnan(x)):
            #     print("NOTE nan encountered after np.where " + str(iod.VR.PaddingValue) + " baseline " + str(baseline) + " gain " + str(gain) + " scale " + str(scale))
            # if np.any(np.isinf(x)):
            #     print("NOTE inf encountered after np.where " + str(iod.VR.PaddingValue) + " baseline " + str(baseline) + " gain " + str(gain) + " scale " + str(scale))
            # if not np.all(np.isfinite(x)):
            #     print("NOTE non-finite encountered " + channel + " after np.where " + str(float_type(iod.VR.PaddingValue)) + " baseline " + str(baseline) + " gain " + str(gain) + " scale " + str(scale))
                
            samples[chan_id][start_target:end_target] = x.astype(iod.VR.FileDatatype)
        
        for channel in channels.keys():
            if channel in channeldefs.keys():
                continue
                
            # recompute the gain, sensitivity, scale, etc for each channel
            vmin = float_type(mins[channel])
            vmax = float_type(maxs[channel])
            gain = float_type(gains[channel])
                        
            baseline = (vmin + vmax) * float_type(-0.5)
            vg_min = (vmin + baseline ) * gain
            vg_max = (vmax + baseline ) * gain
            vg_mag = np.fmax.reduce([np.abs(vg_min), np.abs(vg_max)])
            # need to leave some room for rounding
            dt_mag = float_type(np.fmin.reduce([np.abs(iod.VR.PaddingValue+ 1), np.abs(np.iinfo(iod.VR.FileDatatype).max)]) - 1)
            # print(channel + "," + str(chunk_id) + " vg_min = " + str(vg_min) + " vg_max = " + str(vg_max) + " baseline = " + str(baseline) + " gain = " + str(gain) + " min = " + str(vmin) + " max = " + str(vmax))
            
            scale = float_type(1.0)
            if (vg_mag != 0) and (dt_mag != 0):
                scale *= (dt_mag / vg_mag)
            
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
            source.CodeValue = iod.channel_coding[channel.upper()]['value']
            source.CodingSchemeDesignator = iod.channel_coding[channel.upper()]['scheme']
            source.CodingSchemeVersion = "unknown"
            source.CodeMeaning = channel
            
            sens = float_type(1.0) / gain
            sensitivity = str(sens)
            if (len(sensitivity) > 16):
                sensitivity = str(np.round(sens, decimals=14))
            chdef.ChannelSensitivity = sensitivity if len(sensitivity) <= 16 else sensitivity[:16]   # gain and ADC resolution goes here
            chdef.ChannelSensitivityUnitsSequence = [Dataset()]
            units = chdef.ChannelSensitivityUnitsSequence[0]
                
            # this also needs a look up from a controlled vocab.  This is not correct here...
            units.CodeValue = unit
            units.CodingSchemeDesignator = "UCUM"
            units.CodingSchemeVersion = "1.4"
            units.CodeMeaning = UCUM_ENCODING[unit]  # this needs to be fixed.
                
            # multiplier to apply to the encoded value to get back the orginal input.
            corr = float_type(1.0) / scale
            correction = str(corr)
            if (len(correction) > 16):
                correction = str(np.round(corr, decimals=14))
            chdef.ChannelSensitivityCorrectionFactor = correction if len(correction) <= 16 else correction[:16]   # gain and ADC resolution goes here
            
            baseln = str(baseline)
            if (len(baseln) > 16):
                if (baseline >= 0.0):
                    baseln = str(np.round(baseline, decimals=14))
                else:
                    baseln = str(np.round(baseline, decimals=13))
            chdef.ChannelBaseline = baseln if len(baseln) <= 16 else baseln[:16]
                        
            chdef.WaveformBitsStored = iod.VR.WaveformBitsAllocated
            # only for amplifier type of AC
            # chdef.FilterLowFrequency = '0.05'
            # chdef.FilterHighFrequency = '300'
            channeldefs[channel] = chdef
            

        wfDS.ChannelDefinitionSequence = channeldefs.values() 
        # actual bytes. arr is a numpy array of shape np.stack((ch1, ch2,...), axis=1)
        # arr = np.stack([ samples[channel] for channel in channel_chunk.keys()], axis=1)
        
        # interleave the samples
        wfDS.WaveformData = np.transpose(samples).tobytes()

        return wfDS

    
    # minmax is dictionary of channel to (min, max) values
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
        
        seq_size = 0
        for group, chanchunks in grouped_channels.items():
            # input to this is [(channel, chunk), ... ]
            multiplexGroupDS = self.create_multiplexed_chunk(waveforms, iod, group, chanchunks,
                                                            start_time=start_time, end_time=end_time)
            if (multiplexGroupDS is not None):
                dataset.WaveformSequence.append(multiplexGroupDS)
                seq_size += 1
        
        # each unique frequency will have a different label for the multiplex group..  The multiplex group will 
        # have a separate instance contain the channels with the same frequency
        # each chunk will have channels with the same frequency (with same label).  Channels for each multiplex group
        # will have the same start and end time/ sample ids.
        if (seq_size > 0):
            return dataset
        else:
            return None
    

# dicom value types are constrained by IOD type
# https://dicom.nema.org/medical/dicom/current/output/chtml/part03/PS3.3.html

