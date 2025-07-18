import pandas as pd

### ---- Load and Prepare Data from NFL Verse GitHub using Years 2016-2024 ---- ###
# Uses Data from Years 2016-2024
years = range(2016,2025)
dfs = []

# Loop through each year to get the data
for year in years:
    url = f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz'
    try:
        df_year = pd.read_csv(url, compression='gzip', low_memory=False)
        dfs.append(df_year)
        print(f"Loaded {year}")
    except Exception as e:
        print(f"Failed to load {year}: {e}")

# Put all the years together 
data = pd.concat(dfs, ignore_index=True)
data.to_csv('data/2016-2024 PBP Data.csv', index=False)


### ---- Preprocess Data for Field Goal Model ---- ###
# Filter for Plays that are Field Goals
data = data.loc[data.play_type == 'field_goal']

# Filter for Kickers with at least 15 attempts in the data to avoid noise from kickers with very few attempts
kicker_attempts = data.groupby('kicker_player_name').size()
data = data[data['kicker_player_name'].isin(kicker_attempts[kicker_attempts >= 15].index)]

# Fill in NaN values for temp and wind for indoor stadiums
data['temp'] = data['temp'].fillna(70)
data['wind'] = data['wind'].fillna(0)
data['surface'] = data['surface'].fillna('grass')

# Save the raw data without removing columnes - Allows me to understand what columns are available for the model
data.to_csv('data/FG Raw Data.csv', index=False)


### ---- Feature Engineering for Model from Raw Data ---- ###
# Add Binary Columns for Stadium Type
data['roof_binary'] = data['roof'].map({
    'outdoors': 0,
    'open': 0,
    'closed': 1,
    'dome': 1
})

# Add Binary Column for Home or Away Team Kicking
data["home_away_binary"] = data["posteam_type"].map({
    "home": 1,
    "away": 0
})

# Add Binary Column for Grass or Turf or Any Other Surface
data['binary_grass_turf'] = data['surface'].map({
    'grass': 1,
    'fieldturf': 0,
    'matrixturf': 0,
    'sportturf': 0,
    'a_turf': 0,
    'astroturf': 0,
    'grass ': 1
})

# Career FG% for each Kicker using cumulative made and attempts (Start from First Kick and Builds from There)
data['cumulative_made'] = (
    data.groupby('kicker_player_name')['success']
    .transform(lambda x: x.cumsum().shift(1).fillna(0))
)

data['cumulative_attempts'] = data.groupby('kicker_player_name')['special'].cumcount()

data['career_fg_pct'] = data['cumulative_made'] / data['cumulative_attempts']
data['career_fg_pct'] = data['career_fg_pct'].replace([float('inf'), float('nan')], 1)

# Clutch Kicks - Binary Column for Kicks in 4th Quarter and time < 5 minutes and score differential between -6 and 6 as well as 2nd quarter with time < 3 minutes and overtime
data['clutch_kick'] = (
    (
        (data['qtr'] == 4) &
        (data['quarter_seconds_remaining'] <= 300) &
        (data['score_differential'].between(-6, 6))
    ) |
    ((data['qtr'] == 2) & (data['quarter_seconds_remaining'] <= 180)
     )|
    (data['qtr'] >= 5)  # Overtime
).astype(int)

                        
# Each Kicker Career Long Field Goal before that kick
data['career_long'] = (
    data.groupby('kicker_player_name')['kick_distance']
    .transform(lambda x: x.expanding().max().shift(1))
)

data['career_long'] = data['career_long'].fillna(40)

# Distance Bins for Kick Distance with FG% for that Bin for each kicker prior to that kick
bins = [0, 29, 39, 49, 59, 80]
labels = ['0-29', '30-39', '40-49', '50-59', '60+']

data['distance_bin'] = pd.cut(data['kick_distance'], bins=bins, labels=labels, right=True)

data['bin_cum_made'] = (
    data.groupby(['kicker_player_name', 'distance_bin'], observed=True)['success']
    .transform(lambda x: x.cumsum().shift(1).fillna(0))
)

data['bin_cum_attempts'] = (
    data.groupby(['kicker_player_name', 'distance_bin'], observed=True)['special']
    .cumcount()
)

data['fg_pct_by_bin_prior'] = data['bin_cum_made'] / data['bin_cum_attempts']
data['fg_pct_by_bin_prior'] = data['fg_pct_by_bin_prior'].fillna(0.5)

# That Season's Field Goal Percentage for each Kicker prior to that kick
data['season_cum_made'] = (
    data.groupby(['kicker_player_name', 'season'], observed=True)['success']
    .transform(lambda x: x.cumsum().shift(1).fillna(0))
)

data['season_cum_attempts'] = (
    data.groupby(['kicker_player_name', 'season'], observed=True)
    .cumcount()
)

data['season_fg_pct'] = data['season_cum_made'] / data['season_cum_attempts']
data['season_fg_pct'] = data['season_fg_pct'].replace([float('inf'), float('nan')], 1)

# Last 5 Kicks FG% for each Kicker (Not Used in the Model - Coefficient was very low)
data['last_5_fg_pct'] = (
    data.groupby(['kicker_player_name', 'season'], observed=True)['success']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())) 

data['last_5_fg_pct'] = data['last_5_fg_pct'].fillna(1)

# Get Percentage of Kicks Made from that distance (League Average FG% by Kick Distance)
fg_pct_by_distance = (
    data.groupby('kick_distance')['success']
    .mean()
    .reset_index()
    .rename(columns={'success': 'fg_pct_by_distance_overall'})
)

# Merge back into the main DataFrame
data = data.merge(fg_pct_by_distance, on='kick_distance', how='left')


# Keep only the columns needed for the model (Allows for Easy Understanding of Data as well as use for later assignments)
columns_to_keep = ["special", "kicker_player_name", "success", "clutch_kick", "career_long", 
                   "cumulative_made", "cumulative_attempts", "career_fg_pct", 
                   "season", "season_cum_made", "season_cum_attempts", "season_fg_pct",
                   "last_5_fg_pct",
                   "distance_bin", "fg_pct_by_bin_prior", "bin_cum_made", "bin_cum_attempts",
                   "kick_distance", "fg_pct_by_distance_overall",
                   "roof", "roof_binary", "temp", "wind", 'binary_grass_turf',
                   "posteam_type", "home_away_binary", "score_differential", 
                   "qtr", "quarter_seconds_remaining", "game_seconds_remaining",
                    "kicker_player_id", "stadium_id", "fg_prob", "ep", "posteam"]

fg_features = data[columns_to_keep]


### ---- Save the Processed Data for Modeling ---- ###
fg_features.to_csv('data/FG Data1.csv', index=False)
