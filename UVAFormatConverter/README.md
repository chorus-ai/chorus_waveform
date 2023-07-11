# UVA Format Converter

This directory contains versions of the Universal File Converter maintained by UVA's Center for Advanced Medical Analytics and our research partners. Full documentation is available at [here](https://github.com/Ostrich-Emulators/PreVent#formatconverter). Questions can be directed to Will Ashe ([wa6gz@uvahealth.org](mailto:wa6gz@uvahealth.org)).

## Usage

The included formatconverter.exe is a prebuilt version, compiled for Windows. It accepts the command line arguments as specified in the documentation, though I typically only use flags from the following subset for most applications:
 
- -f or --from [input_filetype]
- -t or --to hdf5 
- -n or --no-break (no break, produces a single file, default is split by day at midnight)
- -p or --pattern [outputpattern] (structured according to documentation)
 
When doing deidentification, I use the following:
 
- --offset [integeroffset] (This shifts the entire time vector, rather than taking a subset)
- -a or --anonymize (I believe this attempts to remove any input filename duplication inside the file as the filename often contains MRNs or datetimes- I don’t think it looks for other PHI)
 
After all flags, the last argument is the file path. Input filename is a single path; for WFDB, this is either a directory containing a single patient’s multisegment file, or the master .hea containing the segment information. Here’s a sample command, where “inputfolder” and “outputfolder” can be any string path, relative or absolute:
 
`formatconverter.exe -f stpxml -t hdf5 -n -p outputfolder/%i.%t inputfolder/inputfilename.xml`
 
This example command takes in the file “inputfolder/inputfilename.xml” (in this case, produced by Bedmaster software), converts to HDF5, creates one output file for the whole input, and then outputs “outputfolder/inputfilename.hdf5”, as the pattern uses %i (input file stem, no directory or extension) and adds the correct extension type with %t (in this case, .hdf5).

## FAQ
### Environment
The prebuilt version is compiled for Windows and runs with only the included libraries. On the [original documentation](https://github.com/Ostrich-Emulators/PreVent#usingbuilding), there are instructions for both compiling on Linux and running with Docker as well as the dependency list, though I haven’t tested either. 
### Sample Files and Scripts
I am working on deidentified examples - check back later.
### Imputation
No imputation is being performed during conversion. Unlike a continuous segment format (i.e. wfdb), the produced data can have gaps, as well as large segments of NaNs/Missing Value Markers. Responsibility for handling these is generally assumed to fall to the end user, who can determine the optimal imputation algorithm for their application.


## CCDEF

This conversion is based off of the CCDEF standard, found [here](https://conduitlab.github.io/ccdef/index.html). We use the ["Single Column Dataset" format](https://conduitlab.github.io/ccdef/datasets.html#single-column-datasets). Known differences:

- No mapping included in current conversion
- Continuous vitals are placed in *VitalSigns* root-level group, rather than *Numerics*

## Known issues

This software is continually revised to update for new libraries and patch bugs. The current bugs under development are:

- No support for FLAC compression when converting from WFDB
- Non-standard attributes (metadata) when converting from different formats
