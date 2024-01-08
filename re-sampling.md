To upsample ECG waveform data from 240Hz to 250Hz in WFDB format, you can use the WFDB Python package along with SciPy or NumPy for resampling. First, ensure you have the `wfdb` and `scipy` packages installed. You can install them using pip:

```bash
pip install wfdb scipy
```

Below is the Python script for the upsampling task:

```python
import wfdb
import numpy as np
from scipy.signal import resample

def resample_ecg(input_file, output_file, original_sr, target_sr):
    # Load the ECG record
    record = wfdb.rdrecord(input_file)
    original_data = record.p_signal

    # Calculate the number of samples in the resampled signal
    num_samples = int((len(original_data) * target_sr) / original_sr)

    # Resample the ECG data
    resampled_data = resample(original_data, num_samples, axis=0)

    # Create a new WFDB record for the resampled data
    resampled_record = wfdb.Record(record_name=output_file,
                                   fs=target_sr,
                                   p_signal=resampled_data,
                                   sig_name=record.sig_name,
                                   units=record.units,
                                   comments=record.comments)

    # Write the resampled record to a file
    wfdb.wrrecord(output_file, resampled_record)

# Example usage
input_file = 'path_to_your_input_file'  # Replace with your input file path (without extension)
output_file = 'path_to_your_output_file'  # Replace with your output file path (without extension)
original_sr = 240
target_sr = 250

resample_ecg(input_file, output_file, original_sr, target_sr)
```

Replace `'path_to_your_input_file'` and `'path_to_your_output_file'` with the actual paths to your input and output files. This script reads the ECG data from the input file, resamples it from 240Hz to 250Hz, and saves it to the output file in WFDB format. Ensure your input file is in a format compatible with the WFDB library.
