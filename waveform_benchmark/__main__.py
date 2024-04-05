import argparse
import csv

from waveform_benchmark.benchmark import run_benchmarks


def read_csv(file_path):
    data = []
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            data.append(row)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_record', '-r',
                    help='The record name to run benchmarking against')
    ap.add_argument('--format_class', '-f',
                    help='The format to save the waveform in')
    ap.add_argument('--physionet_directory', '-p',
                    default=None,
                    help='The physionet database directory to read the source waveform from')
    ap.add_argument('--waveform_suite_table', '-s',
                    default=None,
                    help='A csv table with input_record, physionet directory, and format class for multiple files')
    opts = ap.parse_args()

    # Check conditions based on the parsed arguments
    if not opts.waveform_suite_table:
        # If a waveform suite table is not provided, input_record and format_class must be provided
        if opts.input_record is None or opts.format_class is None:
            ap.error('--input_record and --format_class are required unless --waveform_suite_table is specified')

    # If a table with multiple files is passed we loop through it
    if opts.waveform_suite_table:
        waveform_suite = read_csv(opts.waveform_suite_table)

        for waveform_file in waveform_suite:
            pn_dir = waveform_file[0]
            record = waveform_file[1]
            format = waveform_file[2]
            run_benchmarks(input_record=record,
                           pn_dir=pn_dir,
                           format_class=format)

    # Run benchmarking against a single file
    else:
        run_benchmarks(input_record=opts.input_record,
                       format_class=opts.format_class,
                       pn_dir=opts.physionet_directory)


if __name__ == '__main__':
    main()
