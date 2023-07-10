# UVA Format Converter

This directory contains versions of the Universal File Converter maintained by UVA's Center for Advanced Medical Analytics and our research partners. Full documentation is available at [here](https://github.com/Ostrich-Emulators/PreVent#formatconverter). Questions can be directed to Will Ashe ([wa6gz@uvahealth.org](mailto:wa6gz@uvahealth.org)).

## CCDEF

This conversion is based off of the CCDEF standard, found [here](https://conduitlab.github.io/ccdef/index.html). We use the ["Single Column Dataset" format](https://conduitlab.github.io/ccdef/datasets.html#single-column-datasets). Known differences:

- No mapping included in current conversion
- Continuous vitals are placed in *VitalSigns* root-level group, rather than *Numerics*

## Known issues

This software is continually revised to update for new libraries and patch bugs. The current bugs under development are:

- No support for FLAC compression when converting from WFDB
- Non-standard attributes (metadata) when converting from different formats
