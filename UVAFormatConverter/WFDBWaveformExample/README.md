```python
# viewing example files requires numpy, h5py, and wfdb modules, all available through pip
# subprocess used to repeat conversion
import h5py
import numpy as np
import os
import wfdb
import subprocess
```

# Set Paths, Run Conversion, Open HDF5

Set the following paths - these defaults should work if run from the WFDB example directory in the git repo.


```python
wfdbfolder_path = "3000221"
formatconverter_path = "..\\4.3.11\\formatconverter.exe"
outputfolder = "."
```

You can repeat the conversion of the WFDB raw files into HDF5 using the below code.


```python
# Repeat conversion using local files
formatconverter_params = [formatconverter_path, '-f', 'wfdb', '-t', 'hdf5', '-p', outputfolder+'/'+'%i.%t', wfdbfolder_path]

f= open(os.path.join(outputfolder,"logout.txt"),"w")
completedprocess = subprocess.run(formatconverter_params, stdout=f, stderr=f, timeout=10*3600)
f.close()
```

Open the resulting file using the h5py module.


```python
hdf5file_path = os.path.join(outputfolder,wfdbfolder_path+".hdf5")
myhdf5 = h5py.File(hdf5file_path)
```

# Explore HDF5
Based off of the [h5py documentation](https://docs.h5py.org/en/stable/quick.html) and the [CCDEF documentation](https://conduitlab.github.io/ccdef/index.html).


```python
# Opening generates a HDF5 file object, shown here in read mode
print(myhdf5)
```

    <HDF5 file "3000221.hdf5" (mode r)>
    

HDF5 files are hierarchical with groups (think folders) and datasets (think files). In some interfaces, this is actually represented as a slash-delimited path in the file, similar to file systems. In h5py, the objects are mapped to a tree of nested dictionaries. The keys of a group in the file are the items in that group (i.e. datasets and subgroups). Here, there are three keys in the main group, corresponding to the three groups of UVA-format CCDEF.  


```python
myhdf5.keys()
```




    <KeysViewHDF5 ['Events', 'VitalSigns', 'Waveforms']>



Since each of these are groups, we can access the group using dictionary syntax, and then view the members of that group.


```python
myhdf5['Waveforms'].keys()
```




    <KeysViewHDF5 ['II', 'III']>



Similarly, we can view VitalSigns and Events. VitalSigns is empty, as the WFDB functionality only includes waveforms under the current converter. Similarly, there are no annotations, so Events only contains the Global_Times dataset (see documentation).


```python
print(myhdf5['VitalSigns'].keys())
print(myhdf5['Events'].keys())
```

    <KeysViewHDF5 []>
    <KeysViewHDF5 ['Global_Times']>
    

Waveform and VitalSigns subgroups have two datasets: time and data. These exist separately for each of the recorded signals.


```python
myhdf5['Waveforms']['II'].keys()
```




    <KeysViewHDF5 ['data', 'time']>



Groups and datasets can both have attributes, key-value pairs containing useful metadata. We can view the attrs keys:


```python
myhdf5.attrs.keys()
```




    <KeysViewHDF5 ['Build Number', 'Duration', 'End Date/Time', 'End Time', 'HDF5 Version', 'Layout Version', 'Source Reader', 'Start Date/Time', 'Start Time', 'Timezone']>



Similarly, we can access these using attrs as a dictionary, for the file: 


```python
for mykey in myhdf5.attrs.keys():
    print(f"\"{mykey}\"" + ': ' + str(myhdf5.attrs[mykey]))
```

    "Build Number": 5c8c88ef
    "Duration": 00:03:59
    "End Date/Time": 1970-01-01T02:14:42Z
    "End Time": 8082000
    "HDF5 Version": 1.12.2
    "Layout Version": 4.1.2
    "Source Reader": WFDB
    "Start Date/Time": 1970-01-01T02:10:43Z
    "Start Time": 7843000
    "Timezone": UTC
    

Or for a signal group:


```python
mysigname = "III"
```


```python
for mykey in myhdf5['Waveforms'][mysigname].attrs.keys():
    print(f"\"{mykey}\"" + ': ' + str(myhdf5['Waveforms'][mysigname].attrs[mykey]))
```

    "Data Label": III
    "Duration": 00:03:59
    "End Date/Time": 1970-01-01T02:14:42Z
    "End Time": 8082000
    "Readings Per Sample": 125
    "Sample Period (ms)": 1000
    "Start Date/Time": 1970-01-01T02:10:43Z
    "Start Time": 7843000
    "Timezone": UTC
    "Unit of Measure": mV
    

Or for a dataset in a signal:


```python
for mykey in myhdf5['Waveforms'][mysigname]['data'].attrs.keys():
    print(f"\"{mykey}\"" + ": " + str(myhdf5['Waveforms'][mysigname]['data'].attrs[mykey]))
```

    "Columns": scaled value
    "Max Value": 127.0
    "Min Value": -127.0
    "Missing Value Marker": -32768
    "Note on Min/Max": Min and Max are raw values (not scaled)
    "Note on Scale": To get from a scaled value back to the real value, divide by 10^<scale>
    "Readings Per Sample": 125
    "Sample Period (ms)": 1000
    "Scale": 0
    "Timezone": UTC
    "Unit of Measure": mV
    "wfdb-adcres": 8
    "wfdb-adczero": 0
    "wfdb-baseline": -65
    "wfdb-gain": 128.0
    "wfdb-initval": -128
    "wfdb-spf": 1
    

# Reading Data from HDF5, Comparing to WFDB
Reading from h5py returns h5py dataset objects. These can easily be converted to numpy arrays using an ellipsis index [...]:


```python
mysig = myhdf5['Waveforms'][mysigname]['data']
mydata = myhdf5['Waveforms'][mysigname]['data'][...]
# some dimensional rearranging
mydata = np.squeeze(mydata, 1)
print(mydata)

#convert to float for NAN-scrubbing
mydata = mydata * 1.0
myNANval = mysig.attrs["Missing Value Marker"]
print(myNANval)
mydata[mydata==myNANval] = np.nan

mytime = myhdf5['Waveforms'][mysigname]['time']
```

    [-32768 -32768 -32768 ...    -10    -11    -13]
    -32768
    

The converter preserves the integer values from the WFDB file, however these come with metadata about the sampling process needed to convert back to real values. This process is reveresed here, using the attributes in the file.


```python
mybias = mysig.attrs["wfdb-baseline"]
mygain = mysig.attrs["wfdb-gain"]

print(mybias)
print(mygain)

mydata = (mydata-mybias)/mygain
```

    -65
    128.0
    

Now, we have time and data and can print values of the signal. However, a quick inspection will show that the time and data vectors are different sizes. Specifically, the data vector is 125 times longer. This corresponds to the Readings Per Sample attribute from above, essentially denoting 125 readings per one second sample. This is done to avoid floating point information in the time vector. 


```python
print(np.size(mytime))
print(np.size(mydata))

myReadingsPerSample = mysig.attrs["Readings Per Sample"]

# Print all data, but print the time in the first column every 125 samples.
for i in range (len(mydata)):
    if i % myReadingsPerSample == 0:
        print(f"{mytime[i//myReadingsPerSample]}\t{mydata[i]}")
    else:
        print(f"\t\t{mydata[i]}")
```

### Data omitted - see jupyter notebook

Now, we can load the same record directly from MIMIC, and view the resulting data.


```python
recnum = '3000221'

if recnum[-1] == 'n':
    mysamps, myfields = wfdb.rdsamp(recnum, pn_dir='mimic3wdb/1.0/'+recnum[:2]+'/'+recnum[:-1]+'/')
else:
    mysamps, myfields = wfdb.rdsamp(recnum, pn_dir='mimic3wdb/1.0/'+recnum[:2]+'/'+recnum+'/')

# do the same load locally
# mysamps, myfields = wfdb.rdsamp("3000221/3000221")

print(np.size(mysamps,0))
for samp in mysamps[:,:]:
    print(samp)

```

### Data omitted - see jupyter notebook
