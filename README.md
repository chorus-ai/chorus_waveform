# CHoRUS Waveform Documentation
CHoRUS waveform specification and various conversion scripts for the CHoRUS project. Feel free to add code that converts your waveform format to HDF5, or to an intermediate format that can be used with an existing converter. Recommended organization is one tool per folder, with subfolders for different versions if applicable. You can add a short description below, with more detailed instructions in a readme in the specific folder.

## Table of Contents
1. [Documentation](#documentation)
2. [Installation](#installation)
3. [How to Use](#how-to-use)
4. [Tools](#tools)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)



## Documentation

<details>
<summary>Data Format</summary>
<br/>

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

</details>

<details>
<summary>Core Groups</summary>
<br/>

<small> **/numerics**</small> (vitals)</br>
<small> **/waveforms**</small> (hemodynamics)</br>
<small> **/clinicals**</small> (ehr)</br>
  
 
CHoRUS follows CCDEF core groups: [CCDEF core groups](https://conduitlab.github.io/ccdef/groups.html) 

</details>

<details>
<summary>Waveform and Numerics</summary>
<br/>

Detail the types of datasets, both waveform and numeric, that are part of the project.

</details>

<details>
<summary>Derived Data & Annotations</summary>
<br/>


Any data that is derived from the core datasets and any annotations that might be relevant.

</details>

<details>
<summary>Standard Signal Names</summary>
<br/>


We will provide information on the standard naming convention for signals within the project.
In progress...

</details>




## Installation


First, you need to install `h5py`. You can do this via pip:

```bash
pip install h5py
```



## How to Use

#### 1. Writing data to an HDF5 file:

Let's create an HDF5 file and write some data to it.

```python
import h5py
import numpy as np

# Create a new HDF5 file
f = h5py.File('data.h5', 'w')

# Create a dataset
data = np.random.randn(100, 100)
dset = f.create_dataset("data", data=data)

# Close the file
f.close()
```

#### 2. Reading data from an HDF5 file:

```python
# Open the HDF5 file
f = h5py.File('data.h5', 'r')

# Read data from the file
data = f['group'][:]

# Close the file
f.close()
```

#### 3. Working with groups:

HDF5 supports hierarchical organization, similar to how files are organized in folders in a filesystem.

```python
# Open the file with write mode
f = h5py.File('data.h5', 'a')

# Create a group
grp = f.create_group("my_group")

# Add data to the group
grp.create_dataset("group_data", data=np.random.randn(50, 50))

# Close the file
f.close()
```

#### 4. Attributes:

You can also attach metadata to datasets and groups using attributes:

```python
# Open the file with write mode
f = h5py.File('data.h5', 'a')

# Set an attribute
f['vitals'].attrs['temp'] = 23.5
f['vitals'].attrs['desc'] = "temperature data"

# Close the file
f.close()
```

#### 5. Reading attributes:

```python
# Open the file in read mode
f = h5py.File('data.h5', 'r')

# Read attributes
temp = f['vitals'].attrs['temp']
desc = f['vitals'].attrs['desc']

# Close the file
f.close()
```


#### 6. Organizing Data in Groups:

In HDF5, you can organize datasets within groups, similar to directories in a filesystem. Here's how you can create the "Numerics", "Waveforms", and "Clinical" groups:

```python
# Open the HDF5 file
f = h5py.File('data.h5', 'a')

# Create the 'Numerics' group
numerics_group = f.create_group("Numerics")

# Create the 'Waveforms' group
waveforms_group = f.create_group("Waveforms")

# Create the 'Clinical' group
clinical_group = f.create_group("Clinical")

# Close the file
f.close()
```

#### 7. Adding Channels to the Waveforms Group:

To add channels to the "Waveforms" group:

```python
# Open the HDF5 file
f = h5py.File('data.h5', 'a')
waveforms_group = f['Waveforms']

# Create datasets for each channel with example data
channels = ['ABP', 'ECG', 'Pleth', 'CVP', 'ICP', 'EEG', 'Paw']
for channel in channels:
    data = np.random.randn(1000)  # Example data for 1000 samples
    waveforms_group.create_dataset(channel, data=data)

# Close the file
f.close()
```

#### 8. Optimizing Numerics Channel Structure:

Here's how you can optimize the structure of channels in the "Numerics" group:

```python
import datetime

# Open the HDF5 file
f = h5py.File('data.h5', 'a')
numerics_group = f['Numerics']

# Create datasets for each channel with only values
numeric_channels = ['HR', 'SpO2', 'ABP', 'NIBP', 'RR']

# For demonstration purposes
start_datetime = datetime.datetime.now()
frequency = 1000  # In Hz, meaning data is sampled every millisecond
missing_value_marker = -9999  # Example marker for missing values

for channel in numeric_channels:
    # Store attributes for each channel
    channel_dataset = numerics_group.create_dataset(channel, (1000,), dtype='f')
    channel_dataset.attrs['start_datetime'] = start_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')
    channel_dataset.attrs['frequency'] = frequency
    channel_dataset.attrs['missing_value_marker'] = missing_value_marker
    
    # For demonstration, let's assume we're storing data for 1000 time points
    values = np.random.randn(1000)
    values[::200] = missing_value_marker
    channel_dataset[:] = values

# Close the file
f.close()
```


## Tools
  
  ### [UVA Converter](https://github.com/chorus-ai/waveform/tree/main/UVAFormatConverter)
  Converts between a variety of formats, including inputs of Bedmaster STP XML, wfdb, tdms, dwc, and existing HDF5. Vitals can be specified as a CSV if not included in waveform file. See UVAFormatConverter folder for more detailed information.
  

## Contributing

#### Reporting Issues

Before submitting a pull request, please ensure you've searched for existing issues that may already address the problem you're encountering. If an issue doesn't already exist, you can create a new one:

1. Navigate to the repository's "Issues" tab.
2. Click on "New Issue".
3. Provide a descriptive title for the issue.
4. Fill in the template, detailing the problem, steps to reproduce, expected outcome, and any additional information that might help.
5. Attach any relevant screenshots or logs.
6. Submit the issue.

#### Making Contributions

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch`
3. Make changes and commit: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-branch`
5. Open a pull request

   
#### Contact

To request access to contribution or for further queries: [dbold@emory.edu](mailto:dbold@emory.edu)

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for more details.


## Acknowledgements


in progress...

