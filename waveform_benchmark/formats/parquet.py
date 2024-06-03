import numpy as np
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict
from waveform_benchmark.formats.base import BaseFormat

ROW_GROUP_SIZE_IN_SECONDS = 500

class Parquet(BaseFormat):
    """
    Example format using Parquet with chunked signals with row group size.
    """

    def write_waveforms(self, path, waveforms):
        # Convert each channel into an array with no gaps.

        for name, waveform in waveforms.items():
            length = waveform['chunks'][-1]['end_sample']
            samples = np.empty(length, dtype=np.float32)
            samples[:] = np.nan # fill with nan
            for chunk in waveform['chunks']:
                start = chunk['start_sample']
                end = chunk['end_sample']
                samples[start:end] = chunk['samples']
            
             # Convert samples list to PyArrow array
            samples_array = pa.array(samples)
            
            # Create a PyArrow Table
            table = pa.Table.from_arrays([samples_array], names=[name])
            
            # Prepare metadata, including unit and samples_per_second
            metadata = {
                b'units': waveform["units"].encode(),
                b'samples_per_second': str(waveform["samples_per_second"]).encode()
            }
            
            table = table.replace_schema_metadata(metadata)
            
             # Write to Parquet file with row group size based on ROW_GROUP_SIZE_IN_SECONDS
            dynamic_row_group_size = round(waveform["samples_per_second"] * ROW_GROUP_SIZE_IN_SECONDS)
            file_name = f"{path}_{name}.parquet"

            if self.fmt == 'Compressed':
                pq.write_table(table, file_name, row_group_size=dynamic_row_group_size, compression='gzip')
            else:
                pq.write_table(table, file_name, row_group_size=dynamic_row_group_size)
            
        
    def read_waveforms(self, path, start_time, end_time, signal_names):

        results = {}
        for signal_name in signal_names:
            filepath = f"{path+'_'+signal_name}.parquet"

            try:
                parquet_file = pq.ParquetFile(filepath)
                
                # Metadata extraction
                metadata = parquet_file.metadata.metadata
                samples_per_second = float(metadata[b'samples_per_second'].decode())
                
                # Calculate row groups to read
                row_group_samples = int(samples_per_second * ROW_GROUP_SIZE_IN_SECONDS)

                # Calculate the exact row groups to read
                start_rg_index = round(start_time * samples_per_second) // row_group_samples
                end_rg_index = round(end_time * samples_per_second) // row_group_samples

                # Initialize an empty array for samples
                samples_list = []

                # Directly access each required row group
                for rg_index in range(start_rg_index, end_rg_index + 1):
                    # Check to avoid reading beyond available row groups
                    if rg_index < parquet_file.num_row_groups:
                        row_group = parquet_file.read_row_group(rg_index, columns=[signal_name])
                        samples_list.append(row_group.column(0).to_numpy())

                # Combine arrays from the list
                samples = np.concatenate(samples_list) if samples_list else np.array([])

                # Calculate sample offsets within the concatenated array
                start_sample_offset = round(start_time * samples_per_second) % row_group_samples
                end_sample_offset = start_sample_offset + round((end_time - start_time) * samples_per_second)

                # Slice the samples array to match the exact requested range
                results[signal_name] = samples[start_sample_offset:end_sample_offset]

            except FileNotFoundError:
                print(f"File not found: {filepath}")
                results[signal_name] = np.array([])
            except Exception as e:
                print(f"Error processing {signal_name}: {e}")
                results[signal_name] = np.array([])

        return results

class Parquet_Compressed(Parquet):
    fmt = 'Compressed'

class Parquet_Uncompressed(Parquet):
    fmt = 'Uncompressed'

# class Parquet(BaseFormat):
#     """
#     Example format using Parquet with chunked signals with row group size.
#     """

#     def process_wave(wave):
#         def set_bit(value, bit_index, bit_value):
#             mask = 1 << bit_index
#             value &= ~mask
#             if bit_value:
#                 value |= mask
#             return value 
#         wave = base64.b64decode(wave)
#         binwave = []

#         for i in range(0, len(wave) - 1, 2):
#             t = (wave[i]) + (wave[i+1] * 256)
#             t = set_bit(t, 15, 0) + (-32768) * (t >> 15)
#             binwave.append(int(t))

#         return binwave

#     def reverse_process_wave(binwave):
#         byte_array = bytearray()
#         for t in binwave:
#             lower_byte = t & 0xFF
#             upper_byte = (t >> 8) & 0xFF
#             byte_array.append(lower_byte)
#             byte_array.append(upper_byte)

#         return base64.b64encode(byte_array).decode('utf-8')
    

#     def write_waveforms(self, path, waveforms):
        
#         for name, waveform in waveforms.items():
#             samples_per_second = waveform['samples_per_second']
#             chunk_size_in_samples = round(CHUNK_SIZE_IN_SECONDS * samples_per_second)
#             length = waveform['chunks'][-1]['end_sample']
#             samples = np.empty(length, dtype=np.float32)
#             samples[:] = np.nan # fill with nan
#             for chunk in waveform['chunks']:
#                 start = chunk['start_sample']
#                 end = chunk['end_sample']
#                 samples[start:end] = chunk['samples']

#             # print the samples to see the data in the chunks
#             print(len(samples))
#             # Split the samples into chunks for Parquet
#             chunks = []
#             for i in range(0, length, chunk_size_in_samples):
#                 chunk = samples[i:i + chunk_size_in_samples]
#                 # add to parquet 
#                 chunks.append(chunk)
#             # Convert to Arrow table
#             chunks = pd.DataFrame(chunks)
#             chunks = pa.Table.from_pandas(chunks)
#             # Write to Parquet
#             pq.write_table(chunks, path + name + '.parquet')
#             # Write metadata
#             metadata = {
#                 'samples_per_second': waveform['samples_per_second'],
#                 'chunks': chunk_size_in_samples,
#             }
#             with open(path + name + '.metadata', 'w') as f:
#                 f.write(str(metadata))
#             # print the metadata to see the data in the chunks
#             print(metadata)


#     def read_waveforms(self, path, start_time, end_time, signal_names):
#         results = {}
      
#         for name in signal_names:
#             # Read Parquet file
#             table = pq.read_table(path + name + '.parquet')
#             # Read metadata
#             with open(path + name + '.metadata', 'r') as f:
#                 metadata = eval(f.read())

#             # Filter data based on start and end times
#             start_sample = int(start_time * metadata['samples_per_second'])
#             end_sample = int(end_time * metadata['samples_per_second'])
#             start_chunk = start_sample // metadata['chunks']
#             end_chunk = end_sample // metadata['chunks'] + 1

#             # Extract relevant chunks
#             chunks = table.column(0).to_pylist()[start_chunk:end_chunk]

#             # Pad the last chunk if necessary
#             last_chunk_size = end_sample % metadata['chunks']
#             if last_chunk_size != 0:
#                 if chunks:
#                     last_chunk = np.array(chunks[-1])
#                     last_chunk = np.resize(last_chunk, metadata['chunks'])  # Resize last_chunk if needed
#                     last_chunk[:last_chunk_size] = last_chunk[:last_chunk_size]
#                     chunks[-1] = last_chunk
#                 else:
#                     print(f"No chunks found for signal '{name}'")

#             # Store the results
#             results[name] = chunks

#         return results
