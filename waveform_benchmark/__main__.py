import argparse

from waveform_benchmark.benchmark import run_benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_record')
    ap.add_argument('format_class')
    opts = ap.parse_args()

    run_benchmarks(input_record=opts.input_record,
                   format_class=opts.format_class)


if __name__ == '__main__':
    main()
