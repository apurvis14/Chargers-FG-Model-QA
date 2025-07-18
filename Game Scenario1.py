import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

### ---- Load All Data for Game Scenario 1 ---- ###
fg_data = pd.read_csv('data/Full FG Data with Predictions.csv')
all_data = pd.read_csv('data/2016-2024 PBP Data.csv')


### ---- Game Scenario 1: Data Analysis ---- ###
### ---- All Plays Data for 1st EP ---- ###
# Filter All Plays that would account for the Chargers game against the Chiefs (Assuming Temp and Wind Estimates)
all_data = all_data[(all_data['temp'] <= 50) & (all_data['wind'] >= 12)]

# EP vs Yardline on 1st Down for the filtered data
data_1st = all_data[all_data['down'] == 1]
avg_ep_by_yardline_1st = data_1st.groupby('yardline_100')['ep'].mean().reset_index()


### ---- FG Data for FG Prob by Distance and Game Scenario Analysis ---- ###
# Filter out Games with Temp > 50 and Wind < 10 for FG Data to Match Chargers-Chiefs Game Conditions
data_filtered = fg_data[(fg_data['temp'] <= 50) & (fg_data['wind'] >= 12)]

# Create an Average FG Probability by Distance for Each Yardage from Filtered Data
fg_prob_by_distance_filtered = (
    data_filtered.groupby('kick_distance')['predicted_probability']
    .mean()
    .reset_index()
    .rename(columns={'predicted_probability': 'fg_prob_by_distance'})
)

# Create an Average FG Probability by Distance for Each Yardage from All FG Data (Not Filtered)
fg_prob_by_distance = (
    fg_data.groupby('kick_distance')['predicted_probability']
    .mean()
    .reset_index()
    .rename(columns={'predicted_probability': 'fg_prob_by_distance'}))


### ---- Calculate Expected Points for Going for It on 4th Down vs Kicking Field Goal ---- ###
# Find the EP for 1st Down at Each Yardline
ep_1st_down = avg_ep_by_yardline_1st.set_index('yardline_100')['ep']

go_for_it_ep = []

# Go for It on 4th Down with 50% Success Rate
for yl in range(1, 100):  # Yardlines 1-99 (can't go for it in the endzones)
    success_ep = ep_1st_down.get(yl + 3, 0)  # EP if successful at this yardline (accounts for gaining 3 yards at least)
    fail_ep = -ep_1st_down.get(100 - yl, 0)  # EP for opponent if you fail (lose possession at this yardline)

    expected_go_ep = 0.5 * success_ep + 0.5 * fail_ep
    go_for_it_ep.append({'yardline_100': yl, 'expected_go_ep': expected_go_ep})

go_for_it_df = pd.DataFrame(go_for_it_ep)

# Calculate Expected Points for Kicking Field Goal at Each Yardline
fg_ep = []

# Field Goal Attempt Calculation
for yl in range(1, 100):  # Yardlines 1-99
    fg_distance = yl + 17  # Add 17 yards for snap and endzone
    fg_prob = (fg_prob_by_distance_filtered.set_index('kick_distance')['fg_prob_by_distance'].get(fg_distance, 0)) * 0.95

    make_ep = 3 * fg_prob  # 3 points if made
    miss_ep = -ep_1st_down.get(100 - yl - 7, 0)  # Opponent's EP if missed (accounts for spotting after miss)

    expected_fg_ep = make_ep + (1 - fg_prob) * miss_ep
    fg_ep.append({'yardline_100': yl, 'expected_fg_ep': expected_fg_ep})

fg_df = pd.DataFrame(fg_ep)


### --- Determine Threshold Yardlines for Decision Recommendation ---- ###
# Merge DataFrames and Find Expected Points Difference
merged_df = pd.merge(go_for_it_df, fg_df, on='yardline_100')
merged_df['difference'] = merged_df['expected_go_ep'] - merged_df['expected_fg_ep']

# Find the First Yardline Where FG EP is within 0.25 of GO EP after the 10 yardline
fg_target = merged_df[(merged_df['yardline_100'] > 10) & (merged_df['difference'] >= -0.25)].iloc[0]['yardline_100'] - 2
print(f"Field Goal target threshold yardline: {fg_target}")

fg_target_distance = int(fg_target + 17)

# Find the First Yardline Where FG EP is greater than GO EP
red_zone_go_for_it = merged_df[(merged_df['yardline_100'] < 10) & (merged_df['difference'] <= 0)].iloc[0]['yardline_100']
print(f"Red Zone Go For It threshold yardline: {red_zone_go_for_it}")


### ---- Additional Calculations for Final Visual Based on EP Differences ---- ###
# Average Go For It EP and FG EP from 0 to 5 yards
avg_go_ep_0_5 = go_for_it_df[(go_for_it_df['yardline_100'] >= 0) & (go_for_it_df['yardline_100'] <= 5)]['expected_go_ep'].mean()
avg_fg_ep_0_5 = fg_df[(fg_df['yardline_100'] >= 0) & (fg_df['yardline_100'] <= 5)]['expected_fg_ep'].mean()
difference_ep_0_5 = avg_fg_ep_0_5 - avg_go_ep_0_5
avg_both_0_5 = round((avg_fg_ep_0_5 + avg_go_ep_0_5) / 2, 2)
percent_diff_0_5 = (difference_ep_0_5 / abs(avg_go_ep_0_5)) * 100 if avg_go_ep_0_5 != 0 else 0
percent_diff_0_5 = round(percent_diff_0_5, 2)

# Average in EP between FG and Go For It from 5 to 34 yards
avg_go_ep_5_34 = go_for_it_df[(go_for_it_df['yardline_100'] > 5) & (go_for_it_df['yardline_100'] <= 34)]['expected_go_ep'].mean()
avg_fg_ep_5_34 = fg_df[(fg_df['yardline_100'] > 5) & (fg_df['yardline_100'] <= 34)]['expected_fg_ep'].mean()
difference_ep_5_34 = avg_fg_ep_5_34 - avg_go_ep_5_34
percent_diff_5_34 = (difference_ep_5_34 / abs(avg_go_ep_5_34)) * 100 if avg_go_ep_5_34 != 0 else 0
percent_diff_5_34 = round(percent_diff_5_34, 2)

# Average in EP between FG and Go For it from 34 to 37 yards
avg_go_ep_34_37 = go_for_it_df[(go_for_it_df['yardline_100'] > 34) & (go_for_it_df['yardline_100'] <= 37)]['expected_go_ep'].mean()
avg_fg_ep_34_37 = fg_df[(fg_df['yardline_100'] > 34) & (fg_df['yardline_100'] <= 37)]['expected_fg_ep'].mean()
difference_ep_34_37 = avg_fg_ep_34_37 - avg_go_ep_34_37
percent_diff_34_37 = (difference_ep_34_37 / abs(avg_fg_ep_34_37)) * 100 if avg_fg_ep_34_37 != 0 else 0
percent_diff_34_37 = round(percent_diff_34_37, 2)

# Average in EP between FG and Go for it from 37 to 45 yards
avg_go_ep_37_45 = go_for_it_df[(go_for_it_df['yardline_100'] > 37) & (go_for_it_df['yardline_100'] <= 45)]['expected_go_ep'].mean()
avg_fg_ep_37_45 = fg_df[(fg_df['yardline_100'] > 37) & (fg_df['yardline_100'] <= 45)]['expected_fg_ep'].mean()
difference_ep_37_45 = avg_fg_ep_37_45 - avg_go_ep_37_45
percent_diff_37_45 = (difference_ep_37_45 / abs(avg_fg_ep_37_45)) * 100 if avg_fg_ep_37_45 != 0 else 0
percent_diff_37_45 = round(percent_diff_37_45, 2) * -1 # Made it Positive to Help with Explanation

print('Percentage Difference in EP between FG and Go For It from 0 to 5 yards:', percent_diff_0_5)
print('Percentage Difference in EP between FG and Go For It from 5 to 34 yards:', percent_diff_5_34)
print('Percentage Difference in EP between FG and Go For It from 34 to 37 yards:', percent_diff_34_37)
print('Percentage Difference in EP between FG and Go For It from 37 to 45 yards:', percent_diff_37_45)

# Avg Success Probability of Target Yard FG Attempt From Average FG Probability by Distance
fg_target_prob = int(round(fg_prob_by_distance_filtered[fg_prob_by_distance_filtered['kick_distance'] == fg_target_distance]['fg_prob_by_distance'].values[0] * 100, 0))
print(f"Average Success Probability of {fg_target_distance} Yard FG Attempt: {fg_target_prob}%")

# Avg Slope in Success Probability from 51 - 57 Yard FG Attempts
slope_target_57 = (fg_prob_by_distance_filtered[fg_prob_by_distance_filtered['kick_distance'] == 57]['fg_prob_by_distance'].values[0] - fg_prob_by_distance_filtered[fg_prob_by_distance_filtered['kick_distance'] == fg_target_distance]['fg_prob_by_distance'].values[0]) / (57 - 51)
slope_target_57 = round(slope_target_57 * 100, 2)
print(f"Average Drop in Success Probability from 51 to 56 Yard FG Attempts: {slope_target_57}% per yard")


### ---- Plot setup for the Game Scenario #1 Final Visual ---- ###
fig, ax = plt.subplots(figsize=(24, 12))

# Set full field limits to depict what we need
ax.set_xlim(-10, 55)
ax.set_ylim(0, 10.01)
ax.set_yticks([])

# Shade Recommendation Zones for Kicking vs Going for It
ax.axvspan(0, red_zone_go_for_it, color='lightgreen', alpha=0.8, label='Either Option')
ax.axvspan(red_zone_go_for_it, fg_target, color='darkgreen', alpha=0.8, label='Kick Field Goal Zone')
ax.axvspan(fg_target, 37, color='yellow', alpha=0.8, label='Caution Zone')
ax.axvspan(37, 45, color='orange', alpha=0.8, label='Go For It Zone')

# Add Labels for Each Zone at the Bottom of the Field
ax.text(red_zone_go_for_it / 2, 5, 'Either Option\nZone', color='black', fontsize=16, fontweight='bold', ha='center', va='center')
ax.text((red_zone_go_for_it + fg_target) / 2, 5, 'Kick Field Goal\nZone', color='black', fontsize=16, fontweight='bold', ha='center', va='center')
ax.text((fg_target + 37) / 2, 5, 'Caution\nZone', color='black', fontsize=16, fontweight='bold', ha='center', va='center')
ax.text((37 + 45) / 2, 5, 'Go For It\nZone', color='black', fontsize=16, fontweight='bold', ha='center', va='center')

# Black Rectangle to Separate Each Zone
ranges = [
    (0, red_zone_go_for_it),
    (red_zone_go_for_it, fg_target),
    (fg_target, 37),
    (37, 45)
]

for start, end in ranges:
    width = end - start
    rect = patches.Rectangle(
        (start, 0),
        width,
        10,
        fill=False,        
        edgecolor='black',
        linewidth=3
    )
    ax.add_patch(rect)

# Shade endzone the Chargers colors
ax.axvspan(-10, 0, color='#0080C6', alpha=0.5)

# Yard lines every 5 yards from 0 to 60
for x in range(0, 56, 5):
    ax.vlines(x, ymin=0, ymax=10, color='black', linewidth=1, linestyles='--')

# Yard line labels every 10 yards
yard_labels = list(range(10, 56, 10))  # Starts at 10 not 0
yard_line_positions = list(range(10, 56, 10))
ax.set_xticks(yard_line_positions, minor=False)
ax.set_xticklabels(yard_labels, minor=False, fontsize=12)

# Yard Lines Labels for the Zones
minor_ticks = [0, red_zone_go_for_it, fg_target, 37, 45]
minor_labels = ['0', f'{int(red_zone_go_for_it)}', f'{int(fg_target)}', '37', '45']
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(minor_labels, minor=True, fontsize=12, fontweight='bold', color='red')

# Add an X-Axis Label Saying Yardline (Line of Scrimmage)
ax.set_xlabel('Yardline (Line of Scrimmage)', fontsize=14, fontweight='bold')

# Add "CHARGERS" in both endzones
ax.text(-5, 5, 'C H A R G E R S', color='white', fontsize=64, fontweight='bold',
        ha='center', va='center', rotation=90)

# Add Chargers logo at midfield
logo_img = plt.imread('chargers_logo.webp')
ax.imshow(logo_img, extent=(45, 55, 3, 7), aspect='auto', alpha=0.8)

# Title
ax.set_title('Game Scenario #1: Go for It vs Kick Field Goal', fontweight='bold', fontsize=24)

# Add Description Text Box with Explanation at the Bottom
description_text = (
    "Decision-making zones for the Week 17 game at the Chiefs on 4th down based on expected points (EP) from similar historic wind and temperature conditions.\n\n"
    "• The 'Either Option Zone' (0 to {red_zone}) - both Kicking a Field Goal and Going for It yield similar EP (~{avg_both_0_5}), so both options are viable.\n\n"
    "• The 'Kick Field Goal Zone' ({red_zone} to {fg_target}) - Kicking is favored due to higher EP ({percent_diff_5_34}% Increase). The longest field goal being a {fg_target_distance} yard attempt with a predicted success probability of {fg_target_prob}%.\n\n"
    "• The 'Caution Zone' ({fg_target} to 37) indicates the start of a steep drop in EP and Success Prob. ({slope_target_57}% per add'l yard) for Kicking. Lean towards Going for It with wind conditions unless situation warrants a Kick.\n\n"
    "• The 'Go For It Zone' (37 to 45) - Going for It is strongly recommended as it provides higher EP ({percent_diff_37_45}% Increase) than Kicking while still having a positive expected points no matter the outcome."
).format(red_zone=int(red_zone_go_for_it), fg_target=int(fg_target), avg_both_0_5=avg_both_0_5, percent_diff_5_34=percent_diff_5_34, percent_diff_37_45=percent_diff_37_45, fg_target_distance=fg_target_distance, slope_target_57=slope_target_57, fg_target_prob=fg_target_prob)

# Add Background Rectangle for Description
bbox_props = dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="lightgray")
ax.text(22.5, -1.2, description_text, fontsize=13, ha='center', va='top', bbox=bbox_props)

# Add a Black Line That Separates the Description from the Field on Figure
fig.add_artist(plt.Line2D([0.1, 0.9], [0.2, 0.2], color='black', linewidth=2, transform=fig.transFigure))
fig.add_artist(plt.Line2D([0.1, 0.1], [0, 0.2], color='black', linewidth=2, transform=fig.transFigure))
fig.add_artist(plt.Line2D([0.9, 0.9], [0, 0.2], color='black', linewidth=2, transform=fig.transFigure))

# Layout fix
plt.tight_layout()
plt.savefig('Andrew Purvis - Game Scenario 1 (Week 17 FG Target Line).png', dpi=300)


### ---- Additional Visualizations for Context ---- ###
# Plot with Threshold Line
plt.figure(figsize=(10, 6))
plt.plot(go_for_it_df['yardline_100'], go_for_it_df['expected_go_ep'], label='Go for It on 4th Down', color='blue')
plt.plot(fg_df['yardline_100'], fg_df['expected_fg_ep'], label='Attempt Field Goal', color='red')
plt.axvline(x=fg_target, color='orange', linestyle='--', label=f'Field Goal Target: {fg_target} yards')
plt.axvline(x=red_zone_go_for_it, color='green', linestyle='--', label=f'Go For It in Red Zone: {red_zone_go_for_it} yards')
plt.xlim(0, 60)
plt.title('Expected Points of Going for It vs Kicking on 4th Down by Yardline')
plt.xlabel('Yardline (distance from opponent’s endzone)')
plt.ylabel('Expected Points (EP)')
plt.grid(True)
plt.legend()
plt.savefig('Scenario 1 Context Images/EP Go For It vs FG with Thresholds.png', dpi=300)

# Plot FG Probability by Distance for Filtered and Non-Filtered Data
plt.figure(figsize=(10, 6))
plt.plot(fg_prob_by_distance['kick_distance'], fg_prob_by_distance['fg_prob_by_distance'], marker='o')
plt.plot(fg_prob_by_distance_filtered['kick_distance'], fg_prob_by_distance_filtered['fg_prob_by_distance'], marker='o', color='orange')
plt.title('Average FG Probability by Kick Distance (Temp ≤ 50°F & Wind ≥ 12 mph)')
plt.xlabel('Kick Distance (yards)')
plt.ylabel('FG Probability')
plt.grid()
plt.savefig('Scenario 1 Context Images/FG Probability by Distance (Good Weather vs Cold and Windy).png', dpi=300)
plt.show()