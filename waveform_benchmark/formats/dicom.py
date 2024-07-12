import os

import numpy as np
from pydicom import dcmwrite, dcmread
from pydicom.filereader import read_file_meta_info
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom import uid, _storage_sopclass_uids
from pydicom.waveforms.numpy_handler import WAVEFORM_DTYPES
from typing import TYPE_CHECKING, cast

import math
import os.path

import time
from datetime import datetime
import pandas as pd
import pprint

import uuid

import warnings
# warnings.filterwarnings("error")
from pydicom.fileset import FileSet, RecordNode

from pydicom.tag import Tag
from waveform_benchmark.formats.base import BaseFormat
import waveform_benchmark.formats.dcm_utils.dcm_waveform_writer as dcm_writer
import waveform_benchmark.formats.dcm_utils.dcm_waveform_reader as dcm_reader

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


CHANNEL_TO_DICOM_IOD = {
    "I":  dcm_writer.GeneralECGWaveform,
    "II":  dcm_writer.GeneralECGWaveform,
    "III":  dcm_writer.GeneralECGWaveform,
    "V":  dcm_writer.GeneralECGWaveform,
    "V2": dcm_writer.GeneralECGWaveform,
    "V5": dcm_writer.GeneralECGWaveform,
    "AVR":  dcm_writer.GeneralECGWaveform,
    "ECG": dcm_writer.AmbulatoryECGWaveform,
    "PLETH":  dcm_writer.ArterialPulseWaveform,
    "RESP":  dcm_writer.RespiratoryWaveform,
    "RR":  dcm_writer.RespiratoryWaveform,
    "CO2": dcm_writer.RespiratoryWaveform,
    "CVP": dcm_writer.HemodynamicWaveform,
    "CVP1": dcm_writer.HemodynamicWaveform,
    "CVP2": dcm_writer.HemodynamicWaveform,
    "PAP": dcm_writer.HemodynamicWaveform,
    "PA2": dcm_writer.HemodynamicWaveform,
    "ABP": dcm_writer.HemodynamicWaveform,
    "AO": dcm_writer.HemodynamicWaveform,
    "MCL": dcm_writer.GeneralECGWaveform,
    "ICP": dcm_writer.HemodynamicWaveform,
    "F3-M2": dcm_writer.SleepEEGWaveform,
    "F4-M1": dcm_writer.SleepEEGWaveform,
    "C3-M2": dcm_writer.SleepEEGWaveform,
    "C4-M1": dcm_writer.SleepEEGWaveform,
    "O1-M2": dcm_writer.SleepEEGWaveform,
    "O2-M1": dcm_writer.SleepEEGWaveform,
    "E1-M2": dcm_writer.SleepEEGWaveform,
    "CHIN1-CHIN2": dcm_writer.ElectromyogramWaveform,
    "ABD": dcm_writer.RespiratoryWaveform,
    "CHEST": dcm_writer.RespiratoryWaveform,
    "AIRFLOW": dcm_writer.RespiratoryWaveform,
    "SAO2": dcm_writer.ArterialPulseWaveform,
    "AR1": dcm_writer.HemodynamicWaveform,
    "AR2": dcm_writer.HemodynamicWaveform,
    "SPO2": dcm_writer.ArterialPulseWaveform,
}


class BaseDICOMFormat(BaseFormat):
    """
    Abstract class for WFDB signal formats.
    """

    # group channels by iod, then split by start and end times. 
    # a second pass identify different frequences within a group, and split those off into separate chunks.

    # create a pandas data table to facilitate dicom file creation.  output should each be a separate series of chunks for an iod.
    def create_channel_table(self, waveforms) -> pd.DataFrame:
        # For example: waveforms['V5'] -> {'units': 'mV', 'samples_per_second': 360, 'chunks': [{'start_time': 0.0, 'end_time': 1805.5555555555557, 'start_sample': 0, 'end_sample': 650000, 'gain': 200.0, 'samples': array([-0.065, -0.065, -0.065, ..., -0.365, -0.335,  0.   ], dtype=float32)}]}

        data = []
        for channel, wf in waveforms.items():
            if 'chunks' not in wf.keys():
                raise ValueError("Chunks not found in waveform")
            if 'samples_per_second' not in wf.keys():
                raise ValueError("Samples per second not found in waveform")
            if 'units' not in wf.keys():
                raise ValueError("Units not found in waveform")

            freq = wf['samples_per_second']
            iod = CHANNEL_TO_DICOM_IOD[channel.upper()]
            group = iod.channel_coding[channel.upper()]['group']

            for i, chunk in enumerate(wf['chunks']):
                if 'start_sample' not in chunk.keys():
                    raise ValueError("Start sample not found in chunk")
                if 'end_sample' not in chunk.keys():
                    raise ValueError("End sample not found in chunk")
                if 'start_time' not in chunk.keys():
                    raise ValueError("Start time not found in chunk")
                if 'end_time' not in chunk.keys():
                    raise ValueError("End time not found in chunk")
                if 'gain' not in chunk.keys():
                    raise ValueError("Gain not found in chunk")
                if 'samples' not in chunk.keys():
                    raise ValueError("Samples not found in chunk")

                # add
                data.append({'channel': channel, 
                             'freq': freq,
                             'chunk_id': -(i + 1),
                             'start_time': chunk['start_time'], 
                             'end_time': chunk['end_time'], 
                            #  'start_sample': chunk['start_sample'], 
                            #  'end_sample': chunk['end_sample'],
                             'iod': iod.__name__,
                             'group': group,
                             'freq_id': None})
                # remove
                data.append({'channel': channel, 
                             'freq': freq,
                             'chunk_id': (i+1),
                             'start_time': chunk['end_time'], 
                             'end_time': chunk['end_time'], 
                            #  'start_sample': chunk['start_sample'], 
                            #  'end_sample': chunk['end_sample'],
                             'iod': iod.__name__,
                             'group': group,
                             'freq_id': None})

        df = pd.DataFrame(data)
        df.sort_values(by=['iod', 'start_time', 'freq', 'chunk_id'], inplace=True)

        # assign freq_id, one per frequency within a group
        sorted = df.groupby(['iod', 'group'])
        out = []
        # first assign a freq_id, which distinguishes the different frequencies within a iod-group.  
        for (iod, group), gr in sorted:

            # within each group, each frequency gets a separate id as we should not have multiple frequencies in a group

            nfreqs = gr['freq'].nunique()
            if nfreqs > 1:
                print("NOTE:  multiple frequencies in a group.  need to split further.")

                freqed = gr.groupby('freq')
                id = 1
                for freq, gr2 in freqed:
                    # now we have a single frequency for the iod/group/time.  label it.
                    gr2['freq_id'] = id
                    id += 1             
                    out.append(gr2.copy())   # modification does not propagate up.   
            else:
                # single freq in group.
                gr['freq_id'] = 1
                out.append(gr.copy())  # modification does not propagate up

        df = pd.concat(out)
        df.sort_values(by=['iod', 'freq_id', 'start_time', 'chunk_id'], inplace=True)
        return df 

    # input:  list of dataframes, each dataframe is a group of channels with same iod, group, and frequency.
    # return: dict, each entry is (iod, group, freq, id) = {start_t, end_t, [{(channel, chunk_id): (s_time, e_time), ...}]}, no overlap in time, all channels in a subchunk are non-zero.
    def split_chunks_temporal_adaptive(self, chunk_table: pd.DataFrame) -> dict:
        # split the chunks so that each is a collection of channels in a group but with same starting and ending time.
        # do not put zeros for missing channels
        # need an ordering of channel starting and ending
        # create new table with time stamp, and chunk id (- for start, + for end)
        # sort in order:  iod, group, freq, timestamp, chunk id

        # now group by iod, freq_id, and time
        chunk_table.sort_values(by=['iod', 'freq_id', 'start_time', 'chunk_id'], inplace=True)

        sorted = chunk_table.groupby(['iod', 'freq_id', 'start_time'])

        # now iterate by time to create teh segments
        out = dict()
        file_id = 0
        last_time = None
        curr_ch_chunk_list = dict()
        for (iod, freq_id, time), gr in sorted:

            # print("GROUPED", iod, freq_id, time, gr)
            # now iterate through the groups and create subchunks
            # since grouped by time, we will have addition and removal of channels/chunks at each time point, but not between.
            # so it's sufficient to use sets to track
            if len(curr_ch_chunk_list) == 0:
                # first subchunk.  no worries.
                for index, ch in gr.iterrows():
                    if ch['chunk_id'] < 0:
                        curr_ch_chunk_list[(ch['channel'], (-ch['chunk_id']) - 1)] = ch['group']
                    elif ch['chunk_id'] > 0:
                        print("WARNING: first subchunk, should not be removing a chunk: ", ch)
                    else:
                        print("ERROR:  chunk id is 0", ch)
            else:
                # not the first subchunk.  save current, then update
                out[(iod, freq_id, file_id)] = {'start_t': last_time, 'end_t': time, 
                                            'channel_chunk': curr_ch_chunk_list.copy() }
                file_id += 1
                for index, ch in gr.iterrows():
                    if ch['chunk_id'] < 0:
                        curr_ch_chunk_list[(ch['channel'], (-ch['chunk_id']) - 1)] = ch['group']
                    elif ch['chunk_id'] > 0:
                        del curr_ch_chunk_list[(ch['channel'], ch['chunk_id'] - 1)]
                    else:
                        print("ERROR:  chunk id is 0", ch)
            last_time = time

            # print(out)    
        # do group by
        return out

    # input:  list of dataframes, each dataframe is a group of channels with same iod, group, and frequency.
    # return: dict, each entry is (iod, group, freq, id) = {start_t, end_t, [{(channel, chunk_id): (s_time, e_time), ...]}.  each chunk is defined as the max span for a set of channels.
    # this is detected via a stack - when stack is empty, a subchunk is created.  note that chunkids are not aligned between channels
    def split_chunks_temporal_merged(self, chunk_table: pd.DataFrame) -> dict:
        # split the chunks so that each is a collection of channels in a group.
        # fill missing channels with zeros
        # need an ordering of channel starting and ending
        # create new table with time stamp, and chunk id (- for start, + for end)
        # sort in order:  iod, group, freq, timestamp, chunk id
        # iterate and push and pop to queue (parenthesis like). create a chunk when queue becomes empty.

        chunk_table.sort_values(by=['iod', 'freq_id', 'start_time', 'chunk_id'], inplace=True)
        sorted = chunk_table.groupby(['iod', 'freq_id'])

        out = dict()
        file_id = 0
        stime = None
        etime = None
        stack = []
        channel_chunk_list = dict()  # store the (channel, chunk_id)
        for (iod, freq_id), df in sorted:
            for index, row in df.iterrows():
                # update first
                etime = row['end_time']
                if row['chunk_id'] < 0:
                    stack.append((row['channel'], (-row['chunk_id']) - 1))  # start of chunk
                    channel_chunk_list[(row['channel'], (-row['chunk_id']) - 1)] = row['group']  # add on insert only
                    # save if needed
                    if len(stack) == 1:
                        # inserted first element in the stack
                        stime = row['start_time']
                elif row['chunk_id'] > 0:
                    stack.remove((row['channel'], row['chunk_id'] - 1))  # end of chunk
                    # update end time on remove only
                    if len(stack) == 0:
                        # everything removed from a stack. this indicates a subchunk is complete
                        if len(channel_chunk_list) > 0:
                            out[(iod, freq_id, file_id)] = {'start_t': stime, 'end_t': etime, 'channel_chunk': channel_chunk_list.copy()}
                            file_id += 1
                        channel_chunk_list = {}  # reset
                        stime = None
                        etime = None
                else:
                    print("ERROR:  chunk id is 0", row)

        return out

    # input:  list of dataframes, each dataframe is a group of channels with same iod, group, and frequency.
    # return: dict, each entry is (iod, group, freq, id) = {start_t, end_t, [(channel, chunk_id), ...]}.  each chunk is a fixed time period.
    # we should first detect the merged chunks then segment the chunks. 
    def split_chunks_temporal_fixed(self, chunk_table: pd.DataFrame, duration_sec: float = 600.0) -> dict:
        # split the chunks so that each is a fixed length with appropriate grouping. some may have partial data
        # need an ordering of channel starting and ending.
        # create new table with time stamp, and chunk id (- for start, + for end)
        # also insert window start times - these form "events" at which to snapshot the channels
        # sort in order:  iod, group, freq, timestamp, chunk id
        # iterate and push and pop to queue (parenthesis like).  when seeing events, create the fixed size chunks.

        # # tried to sorting chunks by timestamps first, then partition.  problem is this could create blocks with no channel signals.        
        # # instead - first get the merged subchunks, then subdivide more
        # sorted = self.split_chunks_temporal_merged(chunk_table)
        # # next process each subchunk:

        chunk_table.sort_values(by=['iod', 'freq_id', 'start_time', 'chunk_id'], inplace=True)
        sorted = chunk_table.groupby(['iod', 'freq_id'])

        # for each subchunk, get the start and end, and insert splitters.
        out = dict()
        file_id = 0
        stime = None
        etime = None
        stack = []
        deleted = set()
        added = dict()  # store the (channel, chunk_id)

        # insert all splitter
        time_series = chunk_table['start_time']
        stime = time_series.min()
        etime = time_series.max()

        splitters = []
        periods = int(math.ceil((etime - stime) / duration_sec))

        for i in range(periods):
            splitter_time = stime + (i+1) * duration_sec

            splitters.append({'channel': "__splitter__", 
                            'freq': -1,
                            'chunk_id': 0,  # note:  this will occur before the end time entries.
                            'start_time': splitter_time, 
                            'end_time': splitter_time,
                            'iod': "any",
                            'group': -1,
                            'freq_id': -1})

        splits = pd.DataFrame(splitters)
        # merge splits and sorted

        for (iod, freq_id), gr in sorted:

            # ====== add splitters
            # df is sorted by time.
            splits_iod = pd.concat([gr, splits]).sort_values(by=['start_time', 'chunk_id'])
            # print(splits_iod)

            # ======= process splits_iod
            stack = []
            added = dict()
            deleted = set()
            splitter_start = stime
            for index, row in splits_iod.iterrows():
                # update first
                if row['chunk_id'] < 0:
                    k = (row['channel'], (-row['chunk_id']) - 1)
                    stack.append(k)  # start of chunk
                    added[k] = row['group']  # add on insert only
                    # save if needed

                elif row['chunk_id'] > 0:
                    k = (row['channel'], row['chunk_id'] - 1)
                    stack.remove(k)  # end of chunk
                    deleted.add(k)
                    # update end time on remove only

                else:  # splitter.
                    # everything removed from a stack. this indicates a subchunk is complete
                    if len(added) > 0:
                        splitter_end = row['start_time']
                        out[(iod, freq_id, file_id)] = {'start_t': splitter_start, 'end_t': splitter_end, 'channel_chunk': added.copy()}
                        file_id += 1
                        splitter_start = splitter_end
                        # update he added list by any queued deletes.
                        for x in deleted:
                            del added[x]
                        deleted = set()

        return out

    def make_iod(self, iod_name: str, hifi: bool, num_channels: int = 1):
        if iod_name == "GeneralECGWaveform":
            return dcm_writer.GeneralECGWaveform(hifi=hifi)
        elif iod_name == "AmbulatoryECGWaveform":
            return dcm_writer.AmbulatoryECGWaveform(hifi=hifi)
        elif iod_name == "SleepEEGWaveform":
            return dcm_writer.SleepEEGWaveform(hifi=hifi)
        elif iod_name == "ElectromyogramWaveform":
            return dcm_writer.ElectromyogramWaveform(hifi=hifi)
        elif iod_name == "ArterialPulseWaveform":
            return dcm_writer.ArterialPulseWaveform(hifi=hifi)
        elif iod_name == "RespiratoryWaveform":
            return dcm_writer.RespiratoryWaveform(hifi=hifi, num_channels=num_channels)
        elif iod_name == "HemodynamicWaveform":
            return dcm_writer.HemodynamicWaveform(hifi=hifi)
        else:
            raise ValueError("Unknown IOD")

    def _pretty_print(self, table: dict):
        for key, value in table.items():
            print(key, ": ", value['start_t'], " ", value['end_t'])
            for k, v in value['channel_chunk'].items():
                print("        ", k, v)
                
    # get channel min and max values, across chunks.
    def _get_waveform_channel_minmax(self, waveforms):
        minmax = {}
        for channel, wf in waveforms.items():
            mins = [ np.nanmin(chunk['samples']) for chunk in wf['chunks'] ]
            maxs = [ np.nanmax(chunk['samples']) for chunk in wf['chunks'] ]
            
            minmax[channel] = (np.nanmin(mins), np.nanmax(maxs))
        return minmax
        
        
    def write_waveforms(self, path, waveforms):
        fs = FileSet()

        # one series per modality
        # as many multiplexed groups as allowed by modality
        # one instance per chunk

        # one dicomdir per study?  or series?
        studyInstanceUID = uid.generate_uid()
        seriesInstanceUID = uid.generate_uid()
        prefix, ext = os.path.splitext(path)
        # import json

        # ======== organize the waveform chunks  (tested)
        channel_table = self.create_channel_table(waveforms)
        # print("TABLE: ", channel_table)
        if (self.chunkSize is None):
            subchunks1 = self.split_chunks_temporal_adaptive(channel_table)
            # print("ADAPTIVE", len(subchunks1))
            # self._pretty_print(subchunks1)
        elif self.chunkSize > 0:            
            subchunks1 = self.split_chunks_temporal_fixed(channel_table, duration_sec = self.chunkSize)
            # print("FIXED", len(subchunks1))
            # self._pretty_print(subchunks1)
        else:
            subchunks1 = self.split_chunks_temporal_merged(channel_table)
            # print("merged", len(subchunks1))
            # self._pretty_print(subchunks1)
    
        minmax = self._get_waveform_channel_minmax(waveforms)    
        #========== now write out =============

        # count channels belonging to respiratory data this is needed for the iod
        count_per_iod = {}
        for channel in waveforms.keys():
            iod_name = CHANNEL_TO_DICOM_IOD[channel.upper()].__name__
            if iod_name not in count_per_iod.keys():
                count_per_iod[iod_name] = 1
            else:
                count_per_iod[iod_name] += 1

        # the format of output of split-chunks
        # { (iod, freq_id, file_id): {start_t, end_t, {(channel, chunk_id) : group, ...}}}
        for (iod_name, freq_id, file_id), chunk_info in subchunks1.items():
            # print("writing ", iod_name, ", ", file_id)

            # create and iod instance
            iod = self.make_iod(iod_name, hifi=self.hifi, num_channels = count_per_iod[iod_name])
   
            # each multiplex group can have its own frequency
            # but if there are different frequencies for channels in a multiplex group, we need to split.

            # create a new file  TO FIX.
            dicom = self.writer.make_empty_wave_filedataset(iod)
            dicom = self.writer.set_order_info(dicom)
            dicom = self.writer.set_study_info(dicom, studyUID = studyInstanceUID, studyDate = datetime.now())
            dicom = self.writer.set_series_info(dicom, iod, seriesUID=seriesInstanceUID)
            dicom = self.writer.set_waveform_acquisition_info(dicom, instanceNumber = file_id)
            dicom = self.writer.add_waveform_chunks_multiplexed(dicom, iod, chunk_info, waveforms, minmax)
            
            # Save DICOM file.  write_like_original is required
            # these the initial path when added - it points to a temp file.
            # instance = fs.add(dicom)

            # Add private tags to the WAVEFORM directory record to help with faster access.
            record = Dataset()
            record.DirectoryRecordType = "WAVEFORM"
            record.ReferencedSOPInstanceUIDInFile = dicom.SOPInstanceUID
            record.ReferencedSOPClassUIDInFile = dicom.SOPClassUID
            record.InstanceNumber = dicom.InstanceNumber
            record.ContentDate = dicom.ContentDate
            record.ContentTime = dicom.ContentTime
            block = record.private_block(0x0099, "CHORUS_AI", create=True)

            # gather metadata from dicom.  to keep as same order, must not use set
            channel_info = []
            group_info =[]
            for seq_id, seq in enumerate(dicom.WaveformSequence):
                freq = seq.SamplingFrequency
                nsample = seq.NumberOfWaveformSamples
                stime = cast(float, seq.MultiplexGroupTimeOffset) / 1000.0
                group_info.append({'freq': freq, 'number_samples': nsample, 'start_time': stime})

                for chan_id, ch in enumerate(seq.ChannelDefinitionSequence):
                    channel_info.append({'channel': ch.ChannelSourceSequence[0].CodeMeaning, 
                                         'group_idx': seq_id,
                                         'channel_idx': chan_id})

            # print("GROUPS: ", group_info)
            # print("CHANNELS: ", channel_info)
            channel_names = [x['channel'] for x in channel_info]
            group_ids = [str(x['group_idx']) for x in channel_info ]
            channel_ids = [str(x['channel_idx']) for x in channel_info]
            freqs = [str(x['freq']) for x in group_info]
            nsamples = [str(x['number_samples']) for x in group_info]
            stimes = [str(x['start_time']) for x in group_info]

            block.add_new(0x21, "LO", ",".join(channel_names)) # all channels   (0099 1021)
            block.add_new(0x22, "LO", ",".join(group_ids)) # group of each channel   (0099 1022)
            block.add_new(0x23, "LO", ",".join(channel_ids)) # id of channel in its group   (0099 1023)

            block.add_new(0x11, "LO", ",".join(freqs)) # group's frequencies   (0099 1011)  
            block.add_new(0x12, "LO", ",".join(nsamples)) # groups' sample count   (0099 1012)   
            block.add_new(0x01, "LO", ",".join(stimes))  # group's start times.  (0099 1013)

            # stime_str = str(min(stimes))
            # stime_str = stime_str if len(stime_str) <= 16 else stime_str[:16]
            # block.add_new(0x01, "DS", stime_str) # file seqs start time   (0099 1001)   // use same VR as MultiplexGroupTimeOffset
            # block.add_new(0x02, "FL", chunk_info['end_t']) # file seqs end time   (0099 1002)

            wave_node = RecordNode(record)

            instance = fs.add_custom(dicom, leaf = wave_node)

            # dcm_path = prefix + "_" + str(file_id) + ext
            # dicom.save_as(dcm_path, write_like_original=False)    

        # ========= and create dicomdir file ====
        # ideally - dicomdir file should have a list of channels inside, and the start and end time stamps.
        # but we may have to keep this in the file metadata field.
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part03/chapter_F.html
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_F.5.24.html

        # fs.write('/mnt/c/Users/tcp19/Downloads/Compressed/test_waveform')
        # fs.copy(path)
        # print(path)
        fs.write(path)
        # copy to a directory with randomly generated name
        # fs.copy('/mnt/c/Users/tcp19/BACKUP/dicom_waveform/' + str(uuid.uuid4()))

    def read_waveforms(self, path, start_time, end_time, signal_names):
        # have to read the whole data set each time if using dcmread.  this is not efficient.

        signal_set = set(signal_names)
        signal_set = {name.upper() : name for name in signal_set}
        # ========== read from dicomdir file
        # ideally - each file should have a list of channels inside, and the start and end time stamps.
        # but we may have to open each file and read to gather that info
        # read as file_set, then use metadata to get the table and see which files need to be accessed.
        # then random access to read.
        t1 = time.time()

        ds = dcmread(path + "/DICOMDIR")

        file_info = {}
        for item in ds.DirectoryRecordSequence:
            # if there is private tag data, use it.

            if (item.DirectoryRecordType == "WAVEFORM") and (Tag(0x0099, 0x1001) in item):

                # keep same order, do not use set
                freqs = [float(x) for x in str.split(item[0x0099, 0x1011].value, sep = ',')]
                samples = [int(x) for x in str.split(item[0x0099, 0x1012].value, sep = ',')]
                stimes = [float(x) for x in str.split(item[0x0099, 0x1001].value, sep = ',')]
                etimes = [x + float(y) / z for x, y, z in zip(stimes, samples, freqs)]

                channels = str.split(item[0x0099, 0x1021].value, sep = ',')
                canonical_channels = [x.upper() for x in channels]
                group_ids = [ int(x) for x in str.split(item[0x0099, 0x1022].value, sep = ',')]
                chan_ids = [int(x) for x in str.split(item[0x0099, 0x1023].value, sep = ',')]

                # get group ids and set of channels for all available channels.
                for (i, chan) in enumerate(channels):
                    group_id = group_ids[i]
                    stime = stimes[group_id]
                    etime = etimes[group_id]

                    # filtering here reduces the number of files to open
                    if (chan.upper() not in signal_set.keys()):
                        continue
                    if (etime <= start_time) or (stime >= end_time):
                        continue

                    # only add key if this is a file to be opened.
                    if item.ReferencedFileID not in file_info.keys():
                        file_info[item.ReferencedFileID] = {}

                    if group_id not in file_info[item.ReferencedFileID].keys():
                        file_info[item.ReferencedFileID][group_id] = []

                    # original channel name
                    channel_info = {'channel': chan,
                                    'channel_idx': chan_ids[i],
                                    'freq': freqs[group_id],
                                    'number_samples': samples[group_id],
                                    'start_time': stime}
                    file_info[item.ReferencedFileID][group_id].append(channel_info)
            else:
                # no metadata, so add mapping of None to indicate need to read metadata from file
                file_info[item.ReferencedFileID] = None

        t2 = time.time()
        d1 = t2 - t1
        t1 = time.time()

        # ========== open specific subfiles and gather the channel and time information as dataframe
        # extract channel, start_time, end_time, start_sample, end_sample, iod, group, freq, freq_id

        # each should use a different padding value.
        output = {}
        info_tags = ["WaveformSequence",
                     "MultiplexGroupTimeOffset",
                        "SamplingFrequency",
                        "WaveformPaddingValue",
                        "ChannelDefinitionSequence",
                        "ChannelSourceSequence",
                        "CodeMeaning",
                        ]
        # file_info contains either None (have to get from individual dicom file), or metadata for matched channel/time
        for file_name, finfo in file_info.items():
            fn = path + "/" + file_name

            read_meta_from_file = (finfo is None)

            # if metadata is in dicomdir, then we have only required files in file_info.
            # if metadata is not in dicomdir, then all files are listed and metadata needs to be retrieved.
            # either way, need to read the file.
            with open(fn, 'rb') as fobj:

                # open the file
                t3 = time.time()
                ds = dcmread(fobj, defer_size = 1000, specific_tags = info_tags)
                seqs_raw = dcm_reader.get_tag(fobj, ds, 'WaveformSequence', defer_size = 1000)
                seqs = cast(list[Dataset], seqs_raw)

                t4 = time.time()
                d5 = t4 - t3

                t3 = time.time()
                arrs = {}
                for group_idx, seq in enumerate(seqs):

                    if read_meta_from_file:                    
                        # get the file metadata (can be saved in DICOMDIR in the future, but would need to change the channel metadata info.)
                        channel_infos = dcm_reader.get_waveform_seq_info(fobj, seq)  # get channel info
                    elif group_idx in finfo.keys():
                        channel_infos = finfo[group_idx]
                    else:
                        # this group in the file is not needed.
                        continue

                    if len(channel_infos) == 0:
                        continue

                    # iterate over the channel_infos now.
                    for info in channel_infos:
                        channel = info['channel'].upper()

                        if (channel not in signal_set.keys()):
                            continue

                        # compute start and end offsets in the file using timestamps
                        freq = float(info['freq'])
                        max_len = int(np.round(end_time * freq)) - int(np.round(start_time * freq))

                        # get multiplex group time window
                        gstart = float(info['start_time'])
                        nsamples = int(info['number_samples'])
                        gend = gstart + float(nsamples) / freq

                        # calculate the intersection of the time window
                        win_start = max(gstart, start_time)
                        win_end = min(gend, end_time)

                        if (win_start >= win_end):
                            # window is not possible 
                            continue
                        # else we have a valid window

                        # compute the start and end offset for the source and destination
                        start_offset = max(0, int(np.round(win_start * freq) - np.round(gstart * freq) ))
                        end_offset = min(nsamples, int(np.round(win_end * freq) - np.round(gstart * freq) ))
                        # compute the start and end offset in the output for this channel
                        target_start = max(0, int(np.round(win_start * freq) - np.round(start_time * freq) ))
                        target_end = min(max_len, int(np.round(win_end * freq) - np.round(start_time * freq) ))

                        # print(" start - end: src time ", gstart, gend, " target time ", start_time, end_time, " freq", freq, " window ", win_start, win_end, " samples : src ", start_offset, end_offset, " target ", target_start, target_end)

                        nsamps = min(end_offset - start_offset,  target_end - target_start)
                        end_offset = start_offset + nsamps
                        target_end = target_start + nsamps

                        if nsamps <= 0:
                            continue
                        # else we have a valid window with positive number of samples

                        # get info about the each channel present.
                        channel_idx = info['channel_idx']
                        requested_channel_name = signal_set[channel]
                        # load the data if never read.  else use cached..
                        if group_idx not in arrs.keys():
                            item = cast(Dataset, seq)
                            arrs[group_idx] = dcm_reader.get_multiplex_array(fobj, item, start_offset, end_offset, as_raw = False)

                        # init the output if not previously allocated
                        if requested_channel_name not in output.keys():
                            output[requested_channel_name] = np.full(shape = max_len, fill_value = np.nan, dtype=np.float64)

                        # copy the data to the output
                        # print("copy ", arrs[group_idx].shape, " to ", output[channel].shape, 
                        #       " from ", target_start, " to ", target_end)
                        output[requested_channel_name][target_start:target_end] = arrs[group_idx][channel_idx, 0:nsamps]

        t2 = time.time()
        d3 = t2 - t1
        # print("time: ", path, " (read, metadata, array) = (", d1, d3, ")")

        # now return output.
        return output


# dicom value types are constrained by IOD type
# https://dicom.nema.org/medical/dicom/current/output/chtml/part03/PS3.3.html


class DICOMHighBits(BaseDICOMFormat):
    # waveform lead names to dicom IOD mapping.   Incomplete.
    # avoiding 12 lead ECG because of the limit in number of samples.

    writer = dcm_writer.DICOMWaveformWriter()
    chunkSize = None # adaptive
    hifi = True


class DICOMLowBits(BaseDICOMFormat):

    writer = dcm_writer.DICOMWaveformWriter()
    chunkSize = None  # adaptive
    hifi = False


class DICOMHighBitsChunked(DICOMHighBits):
    # waveform lead names to dicom IOD mapping.   Incomplete.
    # avoiding 12 lead ECG because of the limit in number of samples.

    chunkSize = 3600.0  # chunk as 1 hr.
    hifi = True


class DICOMLowBitsChunked(DICOMLowBits):

    chunkSize = 3600.0
    hifi = False


class DICOMHighBitsMerged(DICOMHighBits):
    # waveform lead names to dicom IOD mapping.   Incomplete.
    # avoiding 12 lead ECG because of the limit in number of samples.

    chunkSize = -1
    hifi = True


class DICOMLowBitsMerged(DICOMLowBits):

    chunkSize = -1
    hifi = False
