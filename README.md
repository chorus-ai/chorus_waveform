# CHoRUS Waveform Documentation
CHoRUS waveform specification and various conversion scripts for the CHoRUS project. Feel free to add code that converts your waveform format to HDF5, or to an intermediate format that can be used with an existing converter. Recommended organization is one tool per folder, with subfolders for different versions if applicable. You can add a short description below, with more detailed instructions in a readme in the specific folder.

## Table of Contents

- [Data Format](#data-format)
- [Core Groups](#core-groups)
- [Waveform and Numerics](#waveform-and-numerics)
- [Derived Data & Annotations](#derived-data--annotations)
- [Standard Signal Names](#standard-signal-names)
- [Tools](#tools)
- [Examples](#examples)
- [Future Development](#future-development)


## Data Format

  #### Signal Storage
  
  Do we allow? both tabular datasets containing multiple simultaneously recorded signals (e.g., multiple ECG leads) and single parameter datasets. Because the format is self-describing, any of the tools used to read the underlying HDF5 file can accommodate either schema.
  
  #### Timestamp Storage
  
  How do we allow time specification? for different time column formats depending on the nature of the data and the need to optimize storage space. Once again, these formats are clearly identified and allow for seamless reading of the file.
  
     Relative
    
    - Time stored as a float number of seconds starting from 0.
    
     Absolute
    
    - Time stored as a float number of seconds starting from `time_origin`.
    - `time_origin` stored in the metadata as a Datetime formatted string.
    - DatetimeIndex reconstructed from `time_origin` and offsets stored in the dataset.
    
     Implied
    
    - No time column in the dataset.
    - Data points must be at a fixed period.
    - DatetimeIndex or TimedeltaIndex reconstructed from `time_origin` and sample rate.


## Core Groups

CHoRUS follows CCDEF core groups:

<small> **/numerics**</small> (vitals)</br>
<small> **/waveforms**</small> (hemodynamics)</br>
<small> **/clinicals**</small> (ehr)</br>
  
  [CCDEF core groups](https://conduitlab.github.io/ccdef/groups.html) 

## Waveform and Numerics

Detail the types of datasets, both waveform and numeric, that are part of the project.


## Derived Data & Annotations

Any data that is derived from the core datasets and any annotations that might be relevant.


## Standard Signal Names

We will provide information on the standard naming convention for signals within the project.
In progress...

## Tools
  
  ### UVA Converter
  Converts between a variety of formats, including inputs of Bedmaster STP XML, wfdb, tdms, dwc, and existing HDF5. Vitals can be specified as a CSV if not included in waveform file. See UVAFormatConverter folder for more detailed information.
  
  [UVA converter](https://github.com/chorus-ai/waveform/tree/main/UVAFormatConverter)


## Examples

In progress...


## Future Development

In progress...

