import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from waveform_benchmark.formats.base import BaseFormat

ROW_GROUP_SIZE_IN_SECONDS = 5


class Parquet(BaseFormat):
    """
    Example format using Parquet.
    """
    def write_waveforms(self, path, waveforms):
        # Convert each channel into an array with no gaps.

        for name, waveform in waveforms.items():
            length = waveform['chunks'][-1]['end_sample']
            samples = np.empty(length, dtype=np.float32)
            samples[:] = np.nan
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

            # Add metadata to table
            table = table.replace_schema_metadata(metadata)

            # Write to Parquet file with row group size based on ROW_GROUP_SIZE_IN_SECONDS
            dynamic_row_group_size = int(waveform["samples_per_second"] * ROW_GROUP_SIZE_IN_SECONDS)
            print('dynqmic row group size:', dynamic_row_group_size)
            file_name = f"{path}_{name}.parquet"
            pq.write_table(table, file_name, row_group_size=dynamic_row_group_size)

    def read_waveforms(self, path, start_time, end_time, signal_names):
        """
        Read waveforms.
        """
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
                start_rg_index = int(start_time * samples_per_second) // row_group_samples
                end_rg_index = int(end_time * samples_per_second) // row_group_samples

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
                start_sample_offset = int(start_time * samples_per_second) % row_group_samples
                end_sample_offset = start_sample_offset + int((end_time - start_time) * samples_per_second)

                # Slice the samples array to match the exact requested range
                results[signal_name] = samples[start_sample_offset:end_sample_offset]

            except FileNotFoundError:
                print(f"File not found: {filepath}")
                results[signal_name] = np.array([])
            except Exception as e:
                print(f"Error processing {signal_name}: {e}")
                results[signal_name] = np.array([])

        return results
