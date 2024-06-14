import os
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import pandas as pd
import seaborn as sns


matplotlib.use('TkAgg')


def get_distinct_colors(n, cmap):
    colors = plt.cm.get_cmap(cmap, n)
    return colors(np.linspace(0, 1, n))


def get_grouped_colors(n_groups, n_per_group, cmap):
    colors = []
    for i in range(n_groups):
        colors.extend(get_distinct_colors(n_per_group, cmap))
    return colors


def update_labels(old_labels):
    new_labels = []
    for label in old_labels:
        if label == 'AtriumDB':
            new_labels.append('AtriumDB_compressed')
        elif label == 'CCDEF_Compressed':
            new_labels.append('CCDEF_compressed')
        elif label == 'CCDEF_Uncompressed':
            new_labels.append('CCDEF_uncompressed')
        elif label == 'DICOMLowBits':
            new_labels.append('DICOM_uncompressed')
        elif label == 'NPY_Uncompressed':
            new_labels.append('Numpy_uncompressed')
        elif label == 'Parquet_Compressed':
            new_labels.append('Parquet_compressed')
        elif label == 'Parquet_Uncompressed':
            new_labels.append('Parquet_uncompressed')
        elif label == 'WFDBFormat16':
            new_labels.append('WFDB_uncompressed')
        elif label == 'WFDBFormat516':
            new_labels.append('WFDB_compressed')
        elif label == 'Zarr':
            new_labels.append('Zarr_uncompressed')
        else:
            new_labels.append(label)

    return new_labels

file_path = 'waveform_suite_benchmark_summary.csv'
df = pd.read_csv(file_path)
test = '500_all' # 'output_size' 'output_time' '500_all' '500_one'
y_label = 'Time (s)' # 'Time (s)' 'Size (KiB)'
title_nm = 'Read time - 500 blocks x 5s, all channels'
# 'Read time - 1 block x total length, all channels'
# 'Read time - 5 blocks x 500s, all channels'
# 'Read time - 50 blocks x 50s, all channels'
# 'Read time - 500 blocks x 5s, all channels'
# 'Read time - 1 block x total length, one channel'
# 'Read time - 5 blocks x 500s, one channel'
# 'Read time - 50 blocks x 50s, one channel'
# 'Read time - 500 blocks x 5s, one channel'
# 'Time to write file'
# 'Written file size'
save_path = './plots/'
save_bar_legend = True
save_bar = True
save_scatter_legend = True
save_scatter = True
scatter_x = 'samples' # 'num_channels' 'length' 'samples'
scatter_y = 'result'

df = df[df['test']==test]

# # Group by 'format' and unstack
grouped = df.groupby(['format_id', 'waveform_id'])['result'].sum().unstack()
grouped = grouped.drop(index=['AtriumDB', 'NPY', 'DICOMHighBits'])

# mask = grouped.index.str.contains('DICOM')
# grouped = grouped[mask]

# # Create the stacked bar plot
grouped.plot(kind='bar', stacked=True)

n_groups = 5
n_per_group = 10
cmap_per_group = ['Blues', 'Greens', 'Greys', 'Purples', 'Reds']
# cmap_per_group = ['viridis', 'Greens', 'Greys', 'Purples', 'Reds']

unique_colors = []
for cmap_name in cmap_per_group:
    unique_colors.extend(get_distinct_colors(n_per_group, cmap_name))

# Plot the DataFrame as a stacked bar plot with grouped colors
ax = grouped.plot(kind='bar', stacked=True, figsize=(12, 8), color=unique_colors)

# Extract the legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)  # Adjust ncol as needed

# Create a new figure for the legend
fig_legend, ax_legend = plt.subplots(figsize=(8, 6))  # Adjust figsize as needed
ax_legend.legend(handles, labels, loc='center', ncol=2)  # Adjust ncol as needed
ax_legend.axis('off')

# Save the legend figure
if save_bar_legend:
    fig_legend.savefig(f'{save_path}/bar_legend.png')

# Remove the legend from the original plot
ax.get_legend().remove()

# Save the plot without the legend by re-creating the plot
fig, ax_no_legend = plt.subplots(figsize=(12, 8))  # Adjust figsize as needed
grouped.plot(kind='bar', stacked=True, ax=ax_no_legend, legend=False, color=unique_colors)

# Labels
plt.xlabel('Format')
plt.ylabel(y_label)
plt.title(title_nm)

# Update format labels
current_labels = ax.get_xticklabels()
label_list = [x.get_text() for x in current_labels]
new_labels = update_labels(label_list)
ax_no_legend.set_xticklabels(new_labels, rotation=45, ha='right')
plt.tight_layout()  # Adjust layout to fit everything nicely

# Save the plot
if save_bar:
    fig.savefig(f'{save_path}{test}_bar.png')

# Clear the plot
plt.clf()

# Plot read times versus number of channels / length / total number of samples
df_wave_meta = pd.read_csv(os.path.join('..', '..', 'data', 'benchmarking', 'waveform_suite', 'waveform_metadata.csv'))

df_wave_meta.drop(columns=['waveform'])
format_ids_to_drop = ['AtriumDB', 'NPY', 'DICOMHighBits']
df = df[~df['format_id'].isin(format_ids_to_drop)]
df['waveform_id'] = df['waveform_id'].astype('object')
df_merged = pd.merge(df, df_wave_meta, on='waveform_id', how='inner')

# Set up the matplotlib figure and axis for the scatter plot
plt.figure(figsize=(10, 6))

scatter_plot = sns.scatterplot(data=df_merged, x=scatter_x, y=scatter_y, hue='format_id', palette='deep', legend=False)

for format_id in df_merged['format_id'].unique():
    subset = df_merged[df_merged['format_id'] == format_id]
    coef = np.polyfit(subset[scatter_x], subset[scatter_y], 1)
    poly1d_fn = np.poly1d(coef)

    # Plot the linear fit
    plt.plot(subset[scatter_x], poly1d_fn(subset[scatter_x]), label=f'{format_id} fit')

# Add labels and title
if scatter_x == 'length':
    x_label = 'Record Length (s)'
elif scatter_x == 'samples':
    x_label = 'Total Number of Samples'
else:
    x_label = 'Number of Channels'
plt.xlabel(x_label)
plt.ylabel(f'Read Time (s)')
plt.title(f'{title_nm} - vs. {scatter_x.replace("_", " ").title()} with best-fit lines')

# Save the scatter plot without the legend
if save_scatter:
    plt.savefig(f'{save_path}linear_scatter/{test}_{scatter_x}.png', bbox_inches='tight')

# Create a separate figure for the legend
plt.figure(figsize=(4, 2))
handles, labels = scatter_plot.get_legend_handles_labels()
scatter_labels = [x.replace(' fit', '') for x in labels]
new_scatter_labels = update_labels(scatter_labels)
legend_fig = plt.gca()
legend_fig.legend(handles, new_scatter_labels, loc='center')
legend_fig.axis('off')

# Save the legend separately
if save_scatter_legend:
    plt.savefig(f'{save_path}linear_scatter/legend_scatter.png', bbox_inches='tight')

# Clear the plot
plt.clf()
