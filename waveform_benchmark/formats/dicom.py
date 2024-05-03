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

import warnings
# warnings.filterwarnings("error")
from pydicom.fileset import FileSet

from waveform_benchmark.formats.base import BaseFormat
import waveform_benchmark.formats.dcm_waveform_writer as dcm_writer
import waveform_benchmark.formats.dcm_waveform_reader as dcm_reader

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

# TODO: [] organize output
# TODO: [] numpy transpose
# TODO: [] dicomdir write
# TODO: [] dicomdir read
# TODO: [] extract with random access
# TODO: [] merge channels


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

CHANNEL_TO_DICOM_IOD = {
    "I":  dcm_writer.GeneralECGWaveform,
    "II":  dcm_writer.GeneralECGWaveform,
    "III":  dcm_writer.GeneralECGWaveform,
    "V":  dcm_writer.GeneralECGWaveform,
    "aVR":  dcm_writer.GeneralECGWaveform,
    "Pleth":  dcm_writer.ArterialPulseWaveform,
    "Resp":  dcm_writer.RespiratoryWaveform,
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
            iod = CHANNEL_TO_DICOM_IOD[channel]
            group = iod.channel_coding[channel]['group']
            

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

        for (iod, freq_id), gr in sorted:
    
            # ====== add splitters
            
            # df is sorted by time.
            time_series = gr['start_time']
            stime = time_series.min()
            etime = time_series.max()
            
            splitters = []
            periods = math.ceil((etime - stime) / duration_sec)
            
            for i in range(periods):
                splitter_time = stime + (i+1) * duration_sec

                splitters.append({'channel': "__splitter__", 
                             'freq': -1,
                             'chunk_id': 0,  # note:  this will occur before the end time entries.
                             'start_time': splitter_time, 
                             'end_time': splitter_time,
                             'iod': iod,
                             'group': -1,
                             'freq_id': -1})
            
            splits = pd.DataFrame(splitters)
            # merge splits and sorted
            splits = pd.concat([gr, splits]).sort_values(by=['start_time', 'chunk_id'])
            
            # print(splits)
            
            # ======= process splits
            splitter_start = stime
            for index, row in splits.iterrows():
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

    def _pretty_print(self, table: dict):
        for key, value in table.items():
            print(key, ": ", value['start_t'], " ", value['end_t'])
            for k, v in value['channel_chunk'].items():
                print("        ", k, v)
    
        
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
        # print("INPUT pleth", waveforms['Pleth'])
        # print("INPUT resp", waveforms['Resp'])
        channel_table = self.create_channel_table(waveforms)
        # print("TABLE: ", channel_table)
        subchunks1 = self.split_chunks_temporal_adaptive(channel_table)
        # print("ADAPTIVE", len(subchunks1))
        # self._pretty_print(subchunks1)
        # subchunks1 = self.split_chunks_temporal_fixed(channel_table, duration_sec = 60000.0)
        # print("FIXED", len(subchunks1))
        # self._pretty_print(subchunks1)
        # subchunks1 = self.split_chunks_temporal_merged(channel_table)
        # print("merged", len(subchunks1))
        # self._pretty_print(subchunks1)
        
        #========== now write out =============
        
        # the format of output of split-chunks
        # { (iod, group, file_id): {start_t, end_t, {(channel, chunk_id) : group, ...}}}
        for (iod_name, freq_id, file_id), chunk_info in subchunks1.items():
            # print("writing ", iod_name, ", ", file_id)
            
            # create and iod instance
            iod = self.make_iod(iod_name)
                                                
            # each multiplex group can have its own frequency
            # but if there are different frequencies for channels in a multiplex group, we need to split.
            
            # create a new file  TO FIX.
            dicom = self.writer.make_empty_wave_filedataset(iod)
            dicom = self.writer.set_order_info(dicom)
            dicom = self.writer.set_study_info(dicom, studyUID = studyInstanceUID, studyDate = datetime.now())
            dicom = self.writer.set_series_info(dicom, iod, seriesUID=seriesInstanceUID)
            dicom = self.writer.set_waveform_acquisition_info(dicom, instanceNumber = file_id)
            dicom = self.writer.add_waveform_chunks_multiplexed(dicom, iod, chunk_info, waveforms)        
            
            # Save DICOM file.  write_like_original is required
            # these the initial path when added - it points to a temp file.
            instance = fs.add(dicom)

            # dcm_path = prefix + "_" + str(file_id) + ext
            # dicom.save_as(dcm_path, write_like_original=False)    
            # dicom.save_as("/mnt/c/Users/tcp19/Downloads/Compressed/test_waveform.dcm", write_like_original=False)

        # ========= and create dicomdir file ====
        # ideally - dicomdir file should have a list of channels inside, and the start and end time stamps.
        # but we may have to keep this in the file metadata field.
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part03/chapter_F.html
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_F.5.24.html
        
        # fs.write('/mnt/c/Users/tcp19/Downloads/Compressed/test_waveform')
        # fs.copy(path)
        # print(path)
        fs.write(path)
        
    
    def read_waveforms(self, path, start_time, end_time, signal_names):
        # have to read the whole data set each time if using dcmread.  this is not efficient.
        
        # ========== read from dicomdir file
        # ideally - each file should have a list of channels inside, and the start and end time stamps.
        # but we may have to open each file and read to gather that info
        # read as file_set, then use metadata to get the table and see which files need to be accessed.
        # then random access to read.
        t1 = time.time()
    
        ds = dcmread(path + "/DICOMDIR", defer_size = 1000)
        fs = FileSet(ds)
    
        t2 = time.time()
        d1 = t2 - t1
        t1 = time.time()
        
        # ========== open specific subfiles and gather the channel and time information as dataframe
        # extract channel, start_time, end_time, start_sample, end_sample, iod, group, freq, freq_id
        fileinfo = []
        for instance in fs:
            # print("Reading ", instance.path)
            # open the file
            with open(instance.path, 'rb') as fobj:
                t3 = time.time()
                ds = dcmread(fobj, defer_size = 100)
                seqs_raw = dcm_reader.get_tag(fobj, ds, 'WaveformSequence', defer_size = 100)
                
                t4 = time.time()
                d5 = t4 - t3

                t3 = time.time()
                seqs = cast(list[Dataset], seqs_raw)
                for idx, seq in enumerate(seqs):
                    infos = dcm_reader.get_waveform_seq_info(fobj, seq)
                    for info in infos:
                        info['end_time'] = np.round(float(info['start_time']) + float(info['number_samples']) / float(info['freq']), decimals=4)
                        if (info['channel'] in signal_names) and (info['start_time'] <= float(end_time)) and (info['end_time'] >= float(start_time)):
                            info['group_idx'] = idx
                            info['filename'] = instance.path
                            fileinfo.append(info)
                t4 = time.time()
                d6 = t4 - t3
                # print("read 1 file: ", d5, d6)

                # TODO merge with wave reading later...
                
        # create a dataframe
        df = pd.DataFrame(fileinfo)
        # -------- now extract the rows with matching channel namees
        # ------- and the rows overlapping with the target time window
        # df = df[(df['channel'].isin(signal_names)) &
        #         (df['start_time'] <= float(end_time)) &
        #         (df['end_time'] >= float(start_time))]
        # print("Seeking channels:  ", signal_names, " and times ", start_time, "-", end_time  )
        # print(df)  
        
        # ======== now open the files and read the channels directly.
        # for each file/multilex_group combination, convert the start and end time to positions in the multiplex group
        #   look up the sequence id, and retrieve the n-d array.
        # note that this returns all channels in the group in interleaved form
        # so we should group the df first by file and seq_id, retrieve, then get the channel ids and extract.
        # to minimize the number of times we need to random access the file.
        # output to one array per channel, in a dictionary.
        
        t2 = time.time()
        d2 = t2 - t1
        t1 = time.time()

        # each should use a different padding value.
        output = {}
        
        # group the df by file and seq_id
        if len(df) > 0:
            files_df = df.groupby('filename')
            
            
            for file, file_gr in files_df:

                groups = file_gr.groupby('group_idx')
                
                # open the file, and get the waveform sequence object
                with open(file, 'rb') as fobj:
                    ds = dcmread(fobj, defer_size = 100)
                    seqs = dcm_reader.get_tag(fobj, ds, 'WaveformSequence', defer_size = 100)
                    
                    for group_idx, group in groups:
                        
                        # compute start and end offsets in the file using timestamps
                        freq = min(group['freq'])
                        max_len = int(np.round(end_time * freq)) - int(np.round(start_time * freq))
                    
                        gstart = min(group['start_time'])
                        gend = max(group['end_time'])
                        nsamples = max(group['number_samples'])
                        start_offset = 0 if start_time <= gstart else int(np.round((start_time - gstart) * freq))
                        end_offset = nsamples if end_time >= gend else int(np.round((end_time - gstart) * freq))
                        # compute the start and end offset in the output for this channel
                        target_start = 0 if gstart <= start_time else int(np.round((gstart - start_time) * freq))
                        target_end = max_len if end_time <= gend else int(np.round((gend - start_time) * freq))
                        
                        # then read the data and transpose.
                        item = cast(list[Dataset], seqs)[group_idx]
                        arr = dcm_reader.get_multiplex_array(fobj, item, start_offset, end_offset, as_raw = False)
                        
                        # then use the channel ids to get the data, and write to output
                        for index, row in group.iterrows():
                            
                            channel = row['channel']
                            padding_value = row['padding_value']
                            if channel not in output.keys():
                                output[channel] = np.full(shape = max_len, 
                                    fill_value = np.nan,
                                    dtype=np.float64)
                            
                            id = row['channel_idx']
                            out = arr[id, :]
                            # print('target range', target_start, ' ', target_end)
                            # print('target_shape ', output[channel].shape)
                            # print('arr shape', arr.shape)
                            # print('out shape', out.shape)
                            # out = np.reshape(arr[id][:], (1, arr.shape[1]))
                            output[channel][target_start:target_end] = out
            
        t2 = time.time()
        d3 = t2 - t1
        # print("time: (read, metadata, array) = (", d1, d2, d3, ")")


        # now return output.
        return output

        # # %% 
        # out = dwh.get_multiplex_array(open, f, seqs, 0, 1000000, 2000000, as_raw = False)
        # gc.collect()
        # print('multiplex:  RAM memory Bytes used:', psutil.virtual_memory()[3])

        # # requested_channels = set(signal_names)
        
        
        
        # # t1 = time.time()
        # # dicom = dcmread(path, defer_size = 32)
        # # t2 = time.time()
        # # # print("Read time", t2 - t1)
        
        # # results = { name: [] for name in signal_names }
        # # dtype = WAVEFORM_DTYPES[(self.WaveformBitsAllocated, self.WaveformSampleInterpretation)]
        
        # # labels = []
        # # for multiplex_group in dicom.WaveformSequence:
        # #     # check match by channel name, start and end time
            
        # #     t1 = time.time()
        # #     group_channels = set([channel_def.ChannelSourceSequence[0].CodeMeaning for channel_def in multiplex_group.ChannelDefinitionSequence ])
        # #     if (len(requested_channels.intersection(group_channels)) == 0):
        # #         # print("skipped due to channel:", group_channels, requested_channels)
        # #         continue
            
        # #     start_t = multiplex_group.MultiplexGroupTimeOffset / 1000.0
        # #     end_t = start_t + float(multiplex_group.NumberOfWaveformSamples) / float(multiplex_group.SamplingFrequency)
            
        # #     if (start_t >= end_time) or (end_t <= start_time):
        # #         # print("skipped outside range", start_t, end_t, start_time, end_time)
        # #         continue
            
        # #     # inbound.  compute the time:
        # #     chunk_start_t = max(start_t, start_time)
        # #     chunk_end_t = min(end_t, end_time)
            
        # #     # # out of bounds.  so exclude.
        # #     # if (chunk_start_t >= chunk_end_t):
        # #     #     print("skipped 0 legnth group ", chunk_start_t, chunk_end_t)
        # #     #     continue
            
        # #     correction_factors = [channel_def.ChannelSensitivityCorrectionFactor for channel_def in multiplex_group.ChannelDefinitionSequence]
            
        # #     # now get the data
        # #     nchannels = multiplex_group.NumberOfWaveformChannels
        # #     nsamples = multiplex_group.NumberOfWaveformSamples
            
        # #     # compute the global and chunk sample offsets.
        # #     if (chunk_start_t == start_t):
        # #         chunk_start_sample = 0
        # #         global_start_sample = np.round(start_t * float(multiplex_group.SamplingFrequency)).astype(int)
        # #     else: 
        # #         chunk_start_sample = np.round((chunk_start_t - start_t) * float(multiplex_group.SamplingFrequency)).astype(int)
        # #         global_start_sample = np.round(chunk_start_t * float(multiplex_group.SamplingFrequency)).astype(int)
        # #     if (chunk_end_t == end_t):
        # #         chunk_end_sample = multiplex_group.NumberOfWaveformSamples
        # #         global_end_sample = global_start_sample + chunk_end_sample
        # #     else:
        # #         chunk_end_sample = np.round((chunk_end_t - start_t) * float(multiplex_group.SamplingFrequency)).astype(int)
        # #         global_end_sample = global_start_sample + chunk_end_sample
            
        # #     t2 = time.time()
        # #     # print(multiplex_group.MultiplexGroupLabel, chunk_start_sample, chunk_end_sample, "metadata", t2 - t1)
            
        # #     t1 = time.time()
        # #     raw_arr = np.frombuffer(cast(bytes, multiplex_group.WaveformData), dtype=dtype).reshape([nsamples, nchannels])
        # #     t2 = time.time()
        # #     # print(multiplex_group.MultiplexGroupLabel, chunk_start_sample, chunk_end_sample, "get raw_arr", t2 - t1)
            
        # #     # print(raw_arr.shape)
        # #     for i, name in enumerate(group_channels):
        # #         if name not in signal_names:
        # #             continue

        # #         # if name not in results.keys():
        # #         #     # results[name] = {}
        # #         #     # results[name]['chunks'] = []
        # #         #     results[name] = []

        # #         # unit = channel_def.ChannelSensitivityUnitsSequence[0].CodeValue
        # #         # gain = 1.0 / channel_def.ChannelSensitivityCorrectionFactor
        # #         # results[name]['units'] = unit
        # #         # results[name]['samples_per_second'] = multiplex_group.SamplingFrequency
        # #         t1 = time.time()
        # #         mask = (raw_arr[chunk_start_sample:chunk_end_sample, i] == self.PaddingValue)
        # #         arr_i = raw_arr[chunk_start_sample:chunk_end_sample, i].astype(self.MemoryDataType, copy=False) * float(correction_factors[i])             # out of bounds.  so exclude.)
        # #         # arr_i = [ x * float(correction_factors[i]) for x in raw_arr[chunk_start_sample:chunk_end_sample, i] ]
        # #         arr_i[mask] = np.nan
        # #         # arr_i[arr_i <= float(self.PaddingValue) ] = np.nan
        # #         # arr_i = arr_i * float(correction_factors[i])
        # #         t2 = time.time()
        # #         # print("convert ", name, " to float.", t2 - t1, start_t, end_t, start_time, end_time)
                
        # #         # chunk = {'start_time': chunk_start_t, 'end_time': chunk_end_t, 
        # #         #                              'start_sample': global_start_sample, 'end_sample': global_end_sample,
        # #         #                              'gain': gain, 'samples': arr_i}
        # #         # results[name]['chunks'].append(chunk)
        # #         results[name].append(arr_i)
                
        # # for name in results.keys():
        # #     if ( len(results[name]) > 0):
        # #         results[name] = np.concatenate(results[name])

        # return results
        ...

# dicom value types are constrained by IOD type
# https://dicom.nema.org/medical/dicom/current/output/chtml/part03/PS3.3.html


    
class DICOMHighBits(BaseDICOMFormat):
    # waveform lead names to dicom IOD mapping.   Incomplete.
    # avoiding 12 lead ECG because of the limit in number of samples.

    writer = dcm_writer.DICOMWaveformWriter()

    def make_iod(self, iod_name: str):
        if iod_name == "GeneralECGWaveform":
            return dcm_writer.GeneralECGWaveform(hifi = True)
        elif iod_name == "ArterialPulseWaveform":
            return dcm_writer.ArterialPulseWaveform(hifi = True)
        elif iod_name == "RespiratoryWaveform":
            return dcm_writer.RespiratoryWaveform(hifi=16, num_channels=1)
        else:
            raise ValueError("Unknown IOD")

    
class DICOMLowBits(BaseDICOMFormat):

    writer = dcm_writer.DICOMWaveformWriter()
    
    def make_iod(self, iod_name: str):
        if iod_name == "GeneralECGWaveform":
            return dcm_writer.GeneralECGWaveform(hifi=False)
        elif iod_name == "ArterialPulseWaveform":
            return dcm_writer.ArterialPulseWaveform(hifi=False)
        elif iod_name == "RespiratoryWaveform":
            return dcm_writer.RespiratoryWaveform(hifi=False, num_channels=1)
        else:
            raise ValueError("Unknown IOD")

