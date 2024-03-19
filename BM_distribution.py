import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

plt.rc('font', family='Helvetica')
pd.options.mode.chained_assignment = None  # Avoid setting copy warning

path_to_parent_folder = "/path/to/lane_n/"
path_excel = "/path/to/lane_n/raw_data.xlsx"
dead_time_start = 24.51
dead_time_end = 50.58
end_time = None

# Load the data
df = pd.read_excel(path_excel, "BMx_fixing")

# Apply dead time masking
index_dead_time_start = (df['time'] - dead_time_start).abs().idxmin()
index_dead_time_end = (df['time'] - dead_time_end).abs().idxmin()
df.loc[index_dead_time_start:index_dead_time_end, df.columns[1:]] = np.nan  # Set to NaN for dead time

if end_time:
    index_end_time = (df['time'] - end_time).abs().idxmin()
    df.loc[index_end_time:, df.columns[1:]] = np.nan  # Set to NaN beyond end time

# Calculate average BM values for each bead before dead time and filter out bad beads
good_beads = []
bad_beads = []
for column in df.columns[1:]:  # Skip the first column as it is 'time'
    avg_bm = df.loc[:index_dead_time_start, column].mean()
    first_25 = df[column][:25].mean()
    last_25 = df[column][-25:].mean()
    last_25_std = df[column][-25:].std()
    if 70 < first_25 < 130:
        good_beads.append(column)
        # if 20 < last_25 < 130:
        #     good_beads.append(column)

    # if 70 < last_25 < 130:
    #     if 10 < last_25_std < 30:
    #         if 20 < first_25 < 130:
    #             good_beads.append(column)
    else:
        bad_beads.append(column)


# Update the dataframe to only include good beads
df_filtered = df[['time'] + good_beads]

# Calculate the median and exclude values during dead time
median_bm = df_filtered.median(axis=1)  # Calculate the median across beads for each timepoint
median_bm[index_dead_time_start:index_dead_time_end] = np.nan  # Exclude median during dead time

# Setting the size of titles and axis numbers
plt.rcParams['axes.labelsize'] = 18  # x and y labels
plt.rcParams['axes.titlesize'] = 18  # title
plt.rcParams['xtick.labelsize'] = 14  # x axis ticks
plt.rcParams['ytick.labelsize'] = 14  # y axis ticks

font = {'fontname': 'Helvetica',
        'weight': 'bold'}

# Initialize lists to store median values
medians_first_25 = []
medians_mid_25 = []
medians_last_25 = []

# Loop through each bead (each column except the first one which is time)
for bead in df_filtered.columns[1:]:  # Skip the first column as it's 'time'
    # Extract first 25 and last 25 timepoints for this bead
    first_25 = df_filtered[bead][:25]
    mid_25 = df_filtered[bead][725:751]
    last_25 = df_filtered[bead][-25:]

    # Compute median for these subsets
    median_first_25 = first_25.median()
    median_mid_25 = mid_25.median()
    median_last_25 = last_25.median()

    # Store these medians
    medians_first_25.append(median_first_25)
    if median_mid_25 < 130:
        medians_mid_25.append(median_mid_25)
    medians_last_25.append(median_last_25)

# Convert lists to arrays for statistical computation
medians_first_25 = np.array(medians_first_25)
medians_mid_25 = np.array(medians_mid_25)
medians_last_25 = np.array(medians_last_25)

# Calculate mean and standard deviation for the medians of the first and last 25 timepoints
mean_first_25 = np.nanmean(medians_first_25)
std_first_25 = np.nanstd(medians_first_25)
mean_mid_25 = np.nanmean(medians_mid_25)
std_mid_25 = np.nanstd(medians_mid_25)
mean_last_25 = np.nanmean(medians_last_25)
std_last_25 = np.nanstd(medians_last_25)

# Print out the results
print(f"Mean of medians (first 25 timepoints): {mean_first_25:.2f}, STD: {std_first_25:.2f}, n = {len(medians_first_25)}")
print(f"Mean of medians (mid 25 timepoints): {mean_mid_25:.2f}, STD: {std_mid_25:.2f}, n = {len(medians_mid_25)}")
print(f"Mean of medians (last 25 timepoints): {mean_last_25:.2f}, STD: {std_last_25:.2f}, n = {len(medians_last_25)}")

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(5, 4))
fig.subplots_adjust(hspace=0)
plt.tick_params(width=1.5)
font_mag = font_manager.FontProperties(family='Helvetica',
                                       weight='bold',
                                       size=14)

# Plotting each set of medians in different subplots
ax[0].hist(medians_first_25, range=(0, 150), bins=40, color="red", alpha=0.5, label=f'n = {len(medians_first_25)}', edgecolor='white')
# ax[0].set_title('First 25 Timepoints', fontsize=14, fontdict=font)
# ax[0].set_ylabel('Frequency', fontsize=14, fontdict=font)
# ax[0].legend(prop=font_mag, fancybox=False, edgecolor='0')
ax[0].tick_params(width=1.5)
ax[0].set_xlim(0, 150)
# ax[0].set_ylim(0, 12)
# ax[0].set_yticks([0, 4, 8])
ax[0].set_yticklabels(np.array([int(i) for i in ax[0].get_yticks()]), fontdict=font)
# ax[0].yaxis.set_ticklabels([])


ax[1].hist(medians_mid_25, range=(0, 150), bins=40, color="red", alpha=0.5, label=f'n = {len(medians_mid_25)}', edgecolor='white')
# ax[1].set_title('Middle 25 Timepoints', fontsize=14, fontdict=font)
# ax[1].set_ylabel('Frequency', fontsize=14, fontdict=font)
# ax[1].legend(prop=font_mag, fancybox=False, edgecolor='0')
ax[1].tick_params(width=1.5)
ax[1].set_xlim(0, 150)
# ax[1].set_ylim(0, 12)
# ax[1].set_yticks([0, 4, 8])
ax[1].set_yticklabels(np.array([int(i) for i in ax[1].get_yticks()]), fontdict=font)
# ax[1].yaxis.set_ticklabels([])


ax[2].hist(medians_last_25, range=(0, 150), bins=40, color="red", alpha=0.5, label=f'n = {len(medians_last_25)}', edgecolor='white')
# ax[2].set_title('Last 25 Timepoints', fontsize=14, fontdict=font)
# ax[2].set_xlabel('Median BM Value (nm)', fontsize=14, fontdict=font)
# ax[2].set_ylabel('Frequency', fontsize=14, fontdict=font)
# ax[2].legend(prop=font_mag, fancybox=False, edgecolor='0')
ax[2].tick_params(width=1.5)
ax[2].set_xlim(0, 150)
# ax[2].set_ylim(0, 12)
# ax[2].set_yticks([0, 4, 8])
ax[2].set_xticklabels(np.array([int(i) for i in ax[2].get_xticks()]), fontdict=font)
ax[2].set_yticklabels(np.array([int(i) for i in ax[2].get_yticks()]), fontdict=font)
# ax[2].xaxis.set_ticklabels([])
# ax[2].yaxis.set_ticklabels([])

for i in range(0, 3):
    ax[i].spines['top'].set_linewidth(1.5)
    ax[i].spines['bottom'].set_linewidth(1.5)
    ax[i].spines['left'].set_linewidth(1.5)
    ax[i].spines['right'].set_linewidth(1.5)

# fig.supylabel('Frequency', fontsize=18, fontdict=font, weight='bold')
# fig.supxlabel('Brownian Motion (nm)', fontsize=18, fontdict=font, weight='bold')
fig.subplots_adjust(hspace=0)
plt.tight_layout()
fig.subplots_adjust(hspace=0)
plt.savefig(path_to_parent_folder + "BMx_distribution_hist.png", dpi=1200)
plt.show()
