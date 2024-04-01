import argparse

from waveform_benchmark.benchmark import run_benchmarks
from waveform_benchmark.formattest import run_conversion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_record')
    ap.add_argument('format_class')
    ap.add_argument('-c', '--check',
                    action='store_true')
    opts = ap.parse_args()

    if(opts.check):
        run_conversion(input_record=opts.input_record,
                   format_class=opts.format_class)
    else:
        run_benchmarks(input_record=opts.input_record,
                   format_class=opts.format_class)          


if __name__ == '__main__':
    main()
