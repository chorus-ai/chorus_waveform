# map of possible file format variations to their shared file extension that
# their respective base script will be called

FILE_FORMAT_MAP = {
    # CRF
    "crf": "crf",

    # PARQUET
    "parquet": "parquet",
    "parq": "parquet",
    "pq": "parquet",

    # CSV
    "csv": "csv",
    "tsv": "csv", # if normalized

    # DICOM
    "dcm": ".dicom",
    "dicom": ".dicom",

    # FEATHER
    "feather": "feather",
    "ft": "feather",

    # HDF5
    "hdf5": "hdf5",
    "hd5": "hdf5",
    "h5": "hdf5",

    # WFDB
    "hea": "wfdb", # header file
    "dat": "wfdb", # signal file

    # sometimes, extension ends in .gz

}