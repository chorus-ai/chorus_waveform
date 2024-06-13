import pandas as pd
import glob

size_list = []
record_list = []
time_list = []
mode_list = []
format_list = []

for filepath in glob.glob('/path/to/results/*/*/*.txt'):
    with open(filepath, "r") as f:
        start = False
        sz = filepath.split('/')[4].split('-')[1]
        record = filepath.split('/')[5]
        format = filepath.split('/')[6].replace('benchmark_results_', '').replace('.txt', '')
        for line in f:
            if start and '___' in line:
                start = False
            if start:
                size_list = size_list + [sz]
                record_list = record_list + [record]
                format_list = format_list + [format]
                str_list = list(filter(None, line.split('  ')))
                if len(str_list) == 5:
                    time_list = time_list + [str_list[3]]
                    mode_list = mode_list + [str_list[4].replace('\n', '')]
                else:
                    time_list = time_list + ['N/A']
                    mode_list = mode_list + ['N/A']
            if "#seek" in line:
                start = True


dict = {'size': size_list,
        'record': record_list,
        'format': format_list,
        'time': time_list,
        'mode': mode_list}
df = pd.DataFrame(dict)
print(df.head())
df.to_csv('performance_processed.csv')
