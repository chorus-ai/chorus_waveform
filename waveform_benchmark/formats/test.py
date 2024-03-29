# %%
from pydicom import dcmread, dcmwrite
from pydicom.data import get_testdata_file
from pydicom.dataset import Dataset
from pydicom import uid
import pydicom._storage_sopclass_uids


# %%
# fpath = get_testdata_file("waveform_ecg.dcm")
# rds = dcmread(fpath)
# rds.SOPClassUID.name
# waveforms = rds.WaveformSequence
# len(waveforms)
# print(waveforms)
# dcmwrite("/mnt/c/Users/tcp19/Downloads/Compressed/waveform_ecg2.dcm", rds )







# %%
import numpy as np
x = np.arange(0, 4 * np.pi, 0.1)
ch1 = (np.cos(x) * (2**15 - 1)).astype('int16')
ch2 = (np.sin(x) * (2**15 - 1)).astype('int16')

# %%
new = Dataset()
# new.MultiplexGroupTimeOffset = '0.0'
new.TriggerTimeOffset = '0.0'
new.WaveformOriginality = "ORIGINAL"
new.NumberOfWaveformChannels = 2
new.NumberOfWaveformSamples = len(x)
new.SamplingFrequency = 1000.0




# %%
new.ChannelDefinitionSequence = [Dataset(), Dataset()]
for i, curve_type in enumerate(["cosine", "sine"]):
    chdef = new.ChannelDefinitionSequence[i]
    # chdef.ChannelTimeSkew = '0'  # Time Skew OR Sample Skew
    chdef.ChannelSampleSkew = "0"
    chdef.WaveformBitsStored = 16
    chdef.ChannelSourceSequence = [Dataset()]
    source = chdef.ChannelSourceSequence[0]
    source.CodeValue = "5.6.3-9-1"
    source.CodingSchemeDesignator = "SCPECG"
    source.CodingSchemeVersion = "1.3"
    source.CodeMeaning = 'Lead I (Einthoven)'
    
    chdef.ChannelSensitivity = '1.0'
    chdef.ChannelSensitivityUnitsSequence = [Dataset()]
    units = chdef.ChannelSensitivityUnitsSequence[0]
    units.CodeValue = "uV"
    units.CodingSchemeDesignator = "UCUM"
    units.CodingSchemeVersion = "1.4"
    units.CodeMeaning = "microvolt"
    
    chdef.ChannelSensitivityCorrectionFactor = '1.0'
    chdef.ChannelBaseline = '0'
    chdef.WaveformBitsStored = 16
    chdef.FilterLowFrequency = '0.05'
    chdef.FilterHighFrequency = '300'
    
    
# %%
arr = np.stack((ch1, ch2), axis=1)
arr.shape
new.WaveformData = arr.tobytes()
new.WaveformBitsAllocated = 16
new.WaveformSampleInterpretation = 'SS'

# %%

fileMeta = Dataset()
fileMeta.MediaStorageSOPClassUID = uid.TwelveLeadECGWaveformStorage if new.NumberOfWaveformChannels == 12 else uid.GeneralECGWaveformStorage
fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

# dataset
ds = Dataset()
ds.file_meta = fileMeta

# Mandatory.  
ds.SOPInstanceUID = uid.generate_uid()
## this should be same as the MediaStorageSOPClassUID
ds.SOPClassUID = fileMeta.MediaStorageSOPClassUID
ds.is_little_endian = True
ds.is_implicit_VR = True

# patient module:
ds.PatientID = "123"
ds.PatientName = "Doe^John"
ds.PatientBirthDate = "19800101"
ds.PatientSex = "M"

# General Study Module
ds.StudyInstanceUID = uid.generate_uid()
ds.ReferringPhysicianName = "Smith^Joseph"
ds.AccessionNumber = "1234"
ds.SeriesInstanceUID = uid.generate_uid()

# General Equipment Module
ds.Manufacturer = "My Company"

# Waveform Identification Module
ds.InstanceNumber = 1
ds.ContentDate = "20200101"
ds.ContentTime = "040000"
ds.AcquisitionDateTime = "20200101040000"

# Acquisition Context Module
ds.AcquisitionContextSequence = [Dataset()]
acqcontext = ds.AcquisitionContextSequence[0]
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

# Content Item Macro module


# needed to build DICOMDIR
ds.StudyDate = "20200101"
ds.StudyTime = "000000"
ds.StudyID = "0"
ds.Modality = "ECG"
ds.SeriesNumber = 0


# waveform data
# ds.NumberOfWaveformChannels = 2
# ds.NumberOfWaveformSamples = len(x)
ds.WaveformSequence = []
ds.WaveformSequence.append(new)



# dcmwrite("/mnt/c/Users/tcp19/Downloads/Compressed/my_waveform.dcm", ds)
# pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)


ds.save_as("/mnt/c/Users/tcp19/Downloads/Compressed/my_waveform.dcm", write_like_original=False)


# %%
from pydicom import dcmread
from matplotlib import pyplot as plt

ds = dcmread("/mnt/c/Users/tcp19/Downloads/Compressed/my_waveform.dcm")
arr = ds.waveform_array(0)
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(arr[:, 0])
ax2.plot(arr[:, 1])
plt.show()

# %%
# 3D image
# from https://stackoverflow.com/questions/14350675/create-pydicom-file-from-numpy-array

# # dummy image
# image = numpy.random.randint(2**16, size=(512, 512, 512), dtype=numpy.uint16)

# # metadata
# fileMeta = Dataset()
# fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
# fileMeta.MediaStorageSOPInstanceUID = uid.generate_uid()
# fileMeta.TransferSyntaxUID = uid.ExplicitVRLittleEndian

# # dataset
# ds = pydicom.Dataset()
# ds.file_meta = fileMeta

# ds.Rows = image.shape[0]
# ds.Columns = image.shape[1]
# ds.NumberOfFrames = image.shape[2]

# ds.PixelSpacing = [1, 1] # in mm
# ds.SliceThickness = 1 # in mm

# ds.BitsAllocated = 16
# ds.PixelRepresentation = 1
# ds.PixelData = image.tobytes()

# # save
# ds.save_as('/mnt/c/Users/tcp19/Downloads/Compressed/fake_image.dcm', write_like_original=False)


# %%
# https://stackoverflow.com/questions/14350675/create-pydicom-file-from-numpy-array

#image
# image2d = numpy.random.randint(2**16, size=(512, 512), dtype=numpy.uint16)

# meta = Dataset()
# meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
# meta.MediaStorageSOPInstanceUID = uid.generate_uid()
# meta.TransferSyntaxUID = uid.ExplicitVRLittleEndian  

# ds = Dataset()
# ds.file_meta = meta

# ds.is_little_endian = True
# ds.is_implicit_VR = False

# ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
# ds.PatientName = "Test^Firstname"
# ds.PatientID = "123456"

# ds.Modality = "MR"
# ds.SeriesInstanceUID = pydicom.uid.generate_uid()
# ds.StudyInstanceUID = pydicom.uid.generate_uid()
# ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

# ds.BitsStored = 16
# ds.BitsAllocated = 16
# ds.SamplesPerPixel = 1
# ds.HighBit = 15

# ds.ImagesInAcquisition = "1"

# ds.Rows = image2d.shape[0]
# ds.Columns = image2d.shape[1]
# ds.InstanceNumber = 1

# ds.ImagePositionPatient = r"0\0\1"
# ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
# ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

# ds.RescaleIntercept = "0"
# ds.RescaleSlope = "1"
# ds.PixelSpacing = r"1\1"
# ds.PhotometricInterpretation = "MONOCHROME2"
# ds.PixelRepresentation = 1

# pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

# print("Setting pixel data...")
# ds.PixelData = image2d.tobytes()

# ds.save_as('/mnt/c/Users/tcp19/Downloads/Compressed/fake_wave2.dcm', write_like_original=False)

# %%


from waveform_benchmark.formats.dicom import DICOMFormat32, DICOMFormat16

dcm32 = DICOMFormat32()
results = dcm32.read_waveforms("/mnt/c/Users/tcp19/Downloads/Compressed/my_waveform.dcm")


# %%
