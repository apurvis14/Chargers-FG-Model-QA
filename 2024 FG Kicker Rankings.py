import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib as matplotlib

### ---- Load Data and Process for 2024 Kicker Rankings ---- ###
# Load FG Data with Predictions and Filter for 2024 Season
data = pd.read_csv('data/Full FG Data with Predictions.csv')
data_2024 = data[data['season'] == 2024]

# Group by kicker and filter for those with at least 10 attempts and create a filtered DF
grouped_2024 = data_2024.groupby('kicker_player_name').size()
valid_kickers_2024 = grouped_2024[grouped_2024 >= 10].index
data_2024_filtered = data_2024[data_2024['kicker_player_name'].isin(valid_kickers_2024)]


### ---- Create Summary DataFrame for 2024 Kicker Rankings ---- ###
# Create Columns for Total Makes, Attempts, and FG% for 2024 Season (Including Postseason) in a New DF
summary_kicks = data_2024_filtered.groupby('kicker_player_name')['success'].agg(
    Makes = lambda x: (x == 1).sum(),
    Attempts = 'count'
) 

# Add The Posteam for Each Kicker (Most Frequent Team)
summary_kicks['Team'] = data_2024_filtered.groupby('kicker_player_name')['posteam'].agg(lambda x: x.mode()[0])

# Add FG% Column
summary_kicks['FG%'] = summary_kicks['Makes'] / summary_kicks['Attempts']

# Create Columnns for Total Makes, Attempts, and FG% at Each Bin Distnace for 2024 Season (Including Postseason)
bin_labels = ['0-29', '30-39', '40-49', '50-59', '60+']
for bin_label in bin_labels:
    makes_col = f'Makes_{bin_label}'
    summary_kicks[makes_col] = data_2024_filtered[
        (data_2024_filtered['distance_bin'] == bin_label) & (data_2024_filtered['success'] == 1)
    ].groupby('kicker_player_name').size()
    summary_kicks[makes_col] = summary_kicks[makes_col].fillna(0).astype(int)

    attempts_col = f'Attempts_{bin_label}'
    summary_kicks[attempts_col] = data_2024_filtered[
        data_2024_filtered['distance_bin'] == bin_label
    ].groupby('kicker_player_name').size()
    summary_kicks[attempts_col] = summary_kicks[attempts_col].fillna(0).astype(int)

    fg_pct_col = f'FG%_{bin_label}'
    summary_kicks[fg_pct_col] = summary_kicks[makes_col] / summary_kicks[attempts_col]
    summary_kicks[fg_pct_col] = summary_kicks[fg_pct_col].fillna(0).astype(float)

# combine 50-59 and 60+ into 50+ bin due to low attempts from above 60
summary_kicks['Makes_50+'] = summary_kicks['Makes_50-59'] + summary_kicks['Makes_60+']
summary_kicks['Attempts_50+'] = summary_kicks['Attempts_50-59'] + summary_kicks['Attempts_60+']

summary_kicks['FG%_50+'] = summary_kicks['Makes_50+'] / summary_kicks['Attempts_50+']
summary_kicks['FG%_50+'] = summary_kicks['FG%_50+'].fillna(0).astype(float)

# Create Column for Clutch Kicks Made, Attempts, and FG% for 2024 Season (Including Postseason)
summary_kicks['Clutch_Makes'] = data_2024_filtered[
    (data_2024_filtered['clutch_kick'] == 1) & (data_2024_filtered['success'] == 1)
].groupby('kicker_player_name').size()

summary_kicks['Clutch_Makes'] = summary_kicks['Clutch_Makes'].fillna(0).astype(int)

summary_kicks['Clutch_Attempts'] = data_2024_filtered[
    data_2024_filtered['clutch_kick'] == 1
].groupby('kicker_player_name').size().reindex(summary_kicks.index, fill_value=0).astype(int)

summary_kicks['Clutch_FG%'] = (summary_kicks['Clutch_Makes'] / summary_kicks['Clutch_Attempts']).fillna(0).astype(float)

# Create Column for Total Expected Points (EP) using Model Predictions
summary_kicks['Expected_Points'] = (data_2024_filtered
                             .assign(expected_points = lambda df: df['predicted_probability'] * 3)
                             .groupby('kicker_player_name')['expected_points'].sum().reindex(summary_kicks.index, fill_value=0).astype(float))

summary_kicks['Actual_Points'] = (data_2024_filtered
                             .assign(actual_points = lambda df: df['success'] * 3)
                             .groupby('kicker_player_name')['actual_points'].sum().reindex(summary_kicks.index, fill_value=0).astype(int))

summary_kicks['Points_OverExpected'] = summary_kicks['Actual_Points'] - summary_kicks['Expected_Points']

# Create a Column for Total Points Over Expected per Attempt
summary_kicks['Points_OE_per_Attempt'] = (summary_kicks['Points_OverExpected'] / summary_kicks['Attempts']).fillna(0).astype(float)

# Create a Percentile Columns for All Relevant Columns that will be used in the Rankings
percentile_columns = ['Makes', 'FG%'] + [f'Makes_{bin}' for bin in bin_labels] + [f'FG%_{bin}' for bin in bin_labels] + ['Clutch_Makes', 'Clutch_FG%'] + ['Points_OverExpected', 'Points_OE_per_Attempt'] + ['Makes_50+', 'FG%_50+']
exclude_cols = ['Makes_50-59', 'Makes_60+', 'Attempts_50-59', 'Attempts_60+']
percentile_columns = [col for col in percentile_columns if col not in exclude_cols]

for col in percentile_columns:
    summary_kicks[f'{col}_Percentile'] = summary_kicks[col].rank(pct=True)

# Create Scoring for Each Category
summary_kicks['FG Makes Score'] = (summary_kicks['Makes_Percentile'] * 0.4 + summary_kicks['FG%_Percentile'] * 0.6) * 100
summary_kicks['FG 0-29 Score'] = (summary_kicks['Makes_0-29_Percentile'] * 0.2 + summary_kicks['FG%_0-29_Percentile'] * 0.8) * 100
summary_kicks['FG 30-39 Score'] = (summary_kicks['Makes_30-39_Percentile'] * 0.2 + summary_kicks['FG%_30-39_Percentile'] * 0.8) * 100
summary_kicks['FG 40-49 Score'] = (summary_kicks['Makes_40-49_Percentile'] * 0.4 + summary_kicks['FG%_40-49_Percentile'] * 0.6) * 100
summary_kicks['FG 50+ Score'] = (summary_kicks['Makes_50+_Percentile'] * 0.4 + summary_kicks['FG%_50+_Percentile'] * 0.6) * 100
summary_kicks['Clutch Kick Score'] = (summary_kicks['Clutch_Makes_Percentile'] * 0.3 + summary_kicks['Clutch_FG%_Percentile'] * 0.7) * 100
summary_kicks['POE/ATT Score'] = summary_kicks['Points_OE_per_Attempt_Percentile'] * 100


### ---- Start the Process of Scoring the Final Rankings ---- ###
# Weights for Overall Ranking Score
weights = {
    'FG Makes Score': 0.16,
    'FG 0-29 Score': 0.08,
    'FG 30-39 Score': 0.08,
    'FG 40-49 Score': 0.16,
    'FG 50+ Score': 0.20,
    'Clutch Kick Score': 0.16,
    'POE/ATT Score': 0.16
}


summary_kicks['Overall Score'] = summary_kicks[list(weights.keys())].mul(list(weights.values())).sum(axis=1)
summary_kicks['Overall Score'] = summary_kicks['Overall Score'].round(2)

# Add Rank Column Based on Overall Score
summary_kicks['Rank'] = summary_kicks['Overall Score'].rank(ascending=False).astype(int)

### ---- Create and Format the Final DataFrame for Display ---- ###
# Rename Kicker Column for Display
df_display = summary_kicks.reset_index()
df_display = df_display.rename(columns={'kicker_player_name': 'Kicker Name'})

# New DF with Weighted Score Columns and Sorted by Rank
final_columns = [
    'Rank', 'Team', 'Kicker Name', 'Overall Score', 'FG Makes Score', 'FG 0-29 Score', 'FG 30-39 Score', 'FG 40-49 Score', 'FG 50+ Score',
    'Clutch Kick Score', 'POE/ATT Score'
]

# Prepare the DataFrame
df_display = df_display[final_columns].sort_values("Rank").reset_index(drop=True)


### ---- Create the Figure Visualization with Table Rankings ---- ###
# Normalize values for color scaling (0 to 100) on score columns
score_cols = final_columns[3:]  # Exclude Rank and Overall Score and Team and Kicker Name
cmap = matplotlib.colormaps.get_cmap('RdYlGn')

# Normalize values for color scale (0 to 100) on Overall Score
norm_overall = matplotlib.colors.Normalize(vmin=0, vmax=100)
cmap_overall = matplotlib.colormaps.get_cmap('BuGn')  # Blue to Green gradient

# Format for display
df_formatted = df_display.copy()
for col in score_cols:
    df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)


# Create figure with Title
fig, ax = plt.subplots(figsize=(28, 16))
fig.suptitle("2024 NFL Kicker Leaderboard", fontsize=24, fontweight='bold', y=0.985)
ax.axis('off')

# Create table with Formatted DataFrame
table = plt.table(
    cellText=df_formatted.values,
    colLabels=df_formatted.columns,
    cellLoc='center',
    bbox=[-0.05, 0.13, 1.05, 0.95]
)

# Resize to ensure fit and readability
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.4, 1.5)

# Change Color the header row
for col_idx in range(len(df_formatted.columns)):
    cell = table[0, col_idx]
    cell.set_facecolor('gray')       # Light gold
    cell.set_linewidth(2.5)
    cell.set_edgecolor('black')
    cell.get_text().set_fontweight('bold')

# Highlight top 5 players (excluding header)
top_n = 5
for row_idx in range(1, top_n + 1):  # skip header (row 0)
    for col_idx in range(len(df_formatted[:3])): # Only 2nd - 3rd columns (Rank, Team, and Kicker Name)
        cell = table[row_idx, col_idx]
        cell.set_facecolor('lightgreen')
        

# Apply conditional color formatting to score columns
for col_name in score_cols:
    col_idx = df_formatted.columns.get_loc(col_name)
    col_values = df_display[col_name]
    col_norm = Normalize(vmin=col_values.min(), vmax=col_values.max())

    for row_idx in range(1, len(df_display) + 1):  # +1 to skip header
        raw_val = df_display.loc[row_idx - 1, col_name]
        color = cmap(col_norm(raw_val))
        hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in color[:3])
        table[row_idx, col_idx].set_facecolor(hex_color)

# Apply conditional color formatting to Overall Score column
overall_col_idx = df_formatted.columns.get_loc('Overall Score')
for row_idx in range(1, len(df_display) + 1):  # +1 to skip header
    raw_val = df_display.loc[row_idx - 1, 'Overall Score']
    color = cmap_overall(norm_overall(raw_val))
    hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in color[:3])
    table[row_idx, overall_col_idx].set_facecolor(hex_color)

# Make First Two Columns Less Wide (Removing Extra White Space)
for col_idx in range(2):  # First two columns (Rank and Team)
    for row_idx in range(len(df_display) + 1):  # +1 to include header
        cell = table[row_idx, col_idx]
        cell.set_width(0.05)  # Set width for Rank and Team columns to be smaller

# Add Legend Box to Explain Each Column
column_explanations = {
    "Rank": "Rank based on Overall Score (1 = best)",
    "Overall Score": "Weighted Composite Score (0-100)",
    "FG Makes Score": "Weighted Score of Total makes & FG% overall (16% of Overall Score)",
    "FG 0-29 Score": "Weighted Score of Makes and FG% from 0-29 yds (8%)",
    "FG 30-39 Score": "Weighted Score of Makes and FG% from 30-39 yds (8%)",
    "FG 40-49 Score": "Weighted Score of Makes and FG% from 40-49 yds (16%)",
    "FG 50+ Score": "Weighted Score of Makes and FG% from 50+ yds (20%)",
    "Clutch Kick Score": "Weighted Score of FGs in Clutch Situations (16%)",
    "POE/ATT Score": "Weighted Score of Points above Expected per Attempt (16%)"
}

# Create two columns for the legend
items = list(column_explanations.items())
half = len(items) // 2 + len(items) % 2
col1 = items[:half]
col2 = items[half:]

# Prepare legend text strings and bold text for the column titles
legend_col1 = "\n".join([rf"$\bf{{{k.replace(' ', r'\ ')}}}$: {v}" for k, v in col1])
legend_col2 = "\n".join([rf"$\bf{{{k.replace(' ', r'\ ')}}}$: {v}" for k, v in col2])

# Add a rectangle for the legend background
legend_x = 0.175
legend_y = 0.05
legend_width = 0.65
legend_height = 0.1
legend_padding_x = 0.005
legend_padding_y = 0.0025

rect = patches.FancyBboxPatch(
    (legend_x, legend_y),
    legend_width, legend_height,
    boxstyle="round,pad=0.02",
    color='lightgrey',
    zorder=0,
    transform=fig.transFigure
)
fig.patches.append(rect)

# Place the legend text with Spacing between each line
fig.text(legend_x + legend_padding_x, legend_y + legend_height - legend_padding_y, legend_col1, fontsize=14, verticalalignment='top', transform=fig.transFigure, family = 'sans-serif', linespacing=2)
fig.text(legend_x + legend_width / 2 + legend_padding_x, legend_y + legend_height - legend_padding_y, legend_col2, fontsize=14, verticalalignment='top', transform=fig.transFigure, family = 'sans-serif', linespacing=2)
fig.text(0.5, legend_y - 0.045, "**All Makes and FG% are Percentiles for each Weighted Score Calculation**", fontsize=16, fontweight='bold', ha='center', transform=fig.transFigure)

# Save or show
plt.savefig("Andrew Purvis - 2024 Kicker Rankings.png", dpi=300, bbox_inches='tight', pad_inches=0.5)