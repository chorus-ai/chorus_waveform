import os
import sys
import pandas as pd
import wfdb
import numpy as np
import random
import sickbay
import sickbay.data
import sickbay.time
import datetime as dt

base_dir = r'PATH_TO_RAW_WAVEFORM_FILES'
out_dir = r'PATH_TO_FINAL_WFDB_FILES'
namespace = 'PHILIPS_PIIC_VITAL' # Namespace variable, needed to identify signal names from the 
#                                   Sickbay Clinical Platform used to collect waveform data at our 
#                                   institution 
pat_deiden_id = random.randint(0, 10000) # Choose a random id for the patient. Adjust as needed.
rshift = random.randint(-60, 60)*24*60*60 # Choose a random datetime shift for the patient. 
#                                           Measured in seconds.

#Grab signal name information from Sickbay platform
#the parquet file stores signal info as integers, and hence we rename the signal ids to something human readable.
#the file contains a url and information about the cookies which correspond to whatever active sickbay session you're logged into in your web browser

#log in to the Sickbay platform using the command line
sickbay.load_session(session_file=r'session_info.json',
                     use_SSL=False)
signal_info = sickbay.data.get_master_signal_list()
signal_info = signal_info[['signal_id', 'namespace', 'display_name', 
                            'class', 'class_name', 'units']].drop_duplicates()
signal_info['signal_id'] = signal_info['signal_id'].astype('int64') 
signal_info = signal_info[signal_info['namespace'] == namespace]

# Lists all waveform files in the base directory. 
# Script assumes these files are saved in parquet format, 
# and that all parquet files in the base directory are waveform files.
file_list = pd.Series(os.listdir(base_dir))
file_list = file_list[file_list.str.contains('.parquet')].reset_index(drop=True)

# The wrdb.wrsamp() function is finicky about the output filename you choose. 
# Setting the script to run in the current directory gets around any complaints
# it may have about special characters in your path
os.chdir(out_dir)
if (not os.path.exists(os.path.join(out_dir, 'wfdb_files'))):
    os.mkdir(os.path.join(out_dir, 'wfdb_files'))

# Convert parquet waveform files into wfdb waveform files
for file in file_list:
    df = pd.read_parquet(os.path.join(base_dir, file))
    # Our waveform data saves the recording timestamp as either the index of a pd.DataFrame 
    # or as a separate column named 'time'
    if ('time' not in df.columns):
        df['time'] = df.index
        df.reset_index(drop=True, inplace=True)
    # Drop any blank columns of signal data
    for col in df.columns:
        isAllNull = len(df[col][~df[col].isna()]) == 0
        if (isAllNull):
            df.drop(col, axis=1, inplace=True) 
    signal_data = df.drop('time', axis=1).values
    fs = round(1/df['time'].diff().mean(

    )) # Hz - this is just based on an average across the file
    first_time = df['time'][0]
    first_time = sickbay.time.epoch_to_local(first_time) #convert epoch time to something readable
    first_time = dt.datetime.strptime(first_time, '%Y-%m-%d %H:%M:%S.%f')
    first_time = first_time + dt.timedelta(seconds = rshift)
    n_signals = signal_data.shape[1]
    col_names = pd.Series(df.columns.to_list())
    col_names = col_names[col_names != 'time'].reset_index(drop=True)
    col_names = col_names.astype('int64') # Raw column names are stored as integers. 
    #                                       Throws an error if we see anything else.
    signal_names = signal_info['class'][signal_info['signal_id'].isin(col_names)].to_list()
    units = signal_info['units'][signal_info['signal_id'].isin(col_names)]
    units[units.isnull()] = 'Unknown'
    units = units.to_list()
    header = {
        'fs': fs,
        'sig_name': signal_names,
        'units': units,
        'comments': ['Converted from pandas DataFrame']
    }
    orig_fname = file.split('.')[0]
    orig_fname_comps = orig_fname.split('_')
    # record_name should be the path to the file where you would like to save the final wfdb file
    record_name = os.path.join('wfdb_files', str(pat_deiden_id) + '_' + 
                                str(int(orig_fname_comps[1]) + rshift) + '_' + 
                                str(int(orig_fname_comps[2]) + rshift) + '_' + 
                                orig_fname_comps[3])
    try:
        wfdb.wrsamp(record_name, fs=fs, units=units, sig_name=signal_names, 
                    p_signal=signal_data, base_time = first_time.time(),
                    base_date=first_time.date(), fmt=['16'] * n_signals)
        print(f"WFDB record '{record_name}' has been created successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

