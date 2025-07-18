import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

### ---- Load Data ---- ###
data = pd.read_csv('data/Full FG Data with Predictions.csv')

### ---- Data Preparation/Analysis ---- ###
# Filter Kickers with at least 15 attempts to Visualize FG Probability for these Kickers
attempts_counts = (data.groupby(['kicker_player_name', 'season']).size().reset_index(name='attempts'))

qualified_kickers = attempts_counts[attempts_counts['attempts'] >= 15]

data1 = data.merge(qualified_kickers[['kicker_player_name', 'season']], on=['kicker_player_name', 'season'], how='inner')

# Calculate FG% per Distance Yardage for Qualified Kickers as well as Average Predicted Probability for each Distance
fg_stats = data1.groupby(['kick_distance']).agg(
    total_attempts=('special', 'sum'),
    total_makes=('success', 'sum'),
    avg_predicted_probability=('predicted_probability', 'mean')
).reset_index()

fg_stats['fg_percentage'] = fg_stats['total_makes'] / fg_stats['total_attempts']

# Filter Kickers with at 15 attempts or less from main data
qualified_kickers2 = attempts_counts[attempts_counts['attempts'] <= 15]

data2 = data.merge(qualified_kickers2[['kicker_player_name', 'season']], on=['kicker_player_name', 'season'], how='inner')

# Calculate FG% per Distance Yardage for less than 15 attempts as well as Average Predicted Probability for each Distance
fg_stats2 = data2.groupby(['kick_distance']).agg(
    total_attempts=('special', 'sum'),
    total_makes=('success', 'sum'),
    avg_predicted_probability=('predicted_probability', 'mean')
).reset_index()

fg_stats2['fg_percentage'] = fg_stats2['total_makes'] / fg_stats2['total_attempts']

# Use Predicted Probability for each Distance and weight it with the FG%
fg_stats['fg_percentage'] = 0.7 * fg_stats['fg_percentage'] + 0.3 * fg_stats['avg_predicted_probability']
fg_stats2['fg_percentage'] = 0.7 * fg_stats2['fg_percentage'] + 0.3 * fg_stats2['avg_predicted_probability']

# Rolling average for smoother curves
fg_stats['fg_percentage'] = fg_stats['fg_percentage'].rolling(window=5, min_periods=1).mean()
fg_stats2['fg_percentage'] = fg_stats2['fg_percentage'].rolling(window=5, min_periods=1).mean()


### --- Merge the two datasets by distance to Find Target Field Goal Distance --- ###
# Compare FG% for both groups for each distance
merged_fg = pd.merge(fg_stats[['kick_distance', 'fg_percentage']], fg_stats2[['kick_distance', 'fg_percentage']], on='kick_distance', suffixes=('_15plus', '_15minus'))

# Target Field Goal Distance - the first distance where 15- attempt FG% is less than 15+ attempt FG% after 40 yards
fg_target = merged_fg[(merged_fg['kick_distance'] > 40) & (merged_fg['fg_percentage_15minus'] < merged_fg['fg_percentage_15plus'])].iloc[0]['kick_distance']
print(f"Field Goal Target Distance: {fg_target}")


### --- Calculate Key Statistics for the Visualization --- ###
# Probability of Making this Field Goal for 15- Attempt Kickers
fg_prob_target = merged_fg[merged_fg['kick_distance'] == fg_target]['fg_percentage_15minus'].values[0]
fg_prob_target = round(fg_prob_target * 100, 0)
print(f"Field Goal Probability at {int(fg_target)} yards: {fg_prob_target:.0f}")

# Find Average FG Probability of Distance from 0 to FG Target for 15- attempt kickers
fg_stats_pre_target = fg_stats2[(fg_stats2['kick_distance'] < fg_target)]
avg_fg_prob_pre_target = round(fg_stats_pre_target['fg_percentage'].mean() * 100, 0)
print(f"Average FG Probability from 0 to {int(fg_target)} yards: {avg_fg_prob_pre_target:.0f}")

# Find Average FG Probability of Distance Past FG Target from 44 - 50 yards for 15- attempt kickers 
fg_stats_post_target = fg_stats2[(fg_stats2['kick_distance'] >= fg_target) & (fg_stats2['kick_distance'] <= 50)]
avg_fg_prob_post_target = round(fg_stats_post_target['fg_percentage'].mean() * 100, 0)
print(f"Average FG Probability from {int(fg_target)} to 50 yards: {avg_fg_prob_post_target:.0f}")

#find Probability of Making 50 Yard Field Goals for 15- Attempt Kickers
fg_stats_50 = fg_stats2[fg_stats2['kick_distance'] == 50]
fg_prob_50 = round(fg_stats_50['fg_percentage'].values[0] * 100, 0)
print(f"FG Probability from 50 yards: {fg_prob_50:.0f}")

# Probability of Making 51+ Yard Field Goals for 15- Attempt Kickers
fg_stats_51_plus = fg_stats2[fg_stats2['kick_distance'] >= 51]
avg_fg_prob_51_plus = round(fg_stats_51_plus['fg_percentage'].mean() * 100, 0)
print(f"Average FG Probability from 51+ yards: {avg_fg_prob_51_plus:.0f}")

# Find Avg FG Probability of Distance from 0 to FG Target for 15+ attempt kickers
fg_stats_pre_target_15plus = fg_stats[(fg_stats['kick_distance'] < fg_target)]
avg_fg_prob_pre_target_15plus = round(fg_stats_pre_target_15plus['fg_percentage'].mean() * 100, 0)
print(f"Average FG Probability from 0 to {int(fg_target)} yards (15+ attempts): {avg_fg_prob_pre_target_15plus:.0f}")

# Find Avg FG Probability of Distance Past FG Target from 44 - 50 yards for 15+ attempt kickers
fg_stats_post_target_15plus = fg_stats[(fg_stats['kick_distance'] >= fg_target) & (fg_stats['kick_distance'] <= 50)]
avg_fg_prob_post_target_15plus = round(fg_stats_post_target_15plus['fg_percentage'].mean() * 100, 0)
print(f"Average FG Probability from {int(fg_target)} to 50 yards (15+ attempts): {avg_fg_prob_post_target_15plus:.0f}")

# Find Avg Probability of Making 51+ Yard Field Goals for 15+ Attempt Kickers
fg_stats_51_plus_15plus = fg_stats[(fg_stats['kick_distance'] >= 51)]
avg_fg_prob_51_plus_15plus = round(fg_stats_51_plus_15plus['fg_percentage'].mean() * 100, 0)
print(f"Average FG Probability from 51+ yards (15+ attempts): {avg_fg_prob_51_plus_15plus:.0f}")

# Find League Difference of These 3 Main Ranges
diff_pre_target = avg_fg_prob_pre_target_15plus - avg_fg_prob_pre_target
diff_post_target = avg_fg_prob_post_target_15plus - avg_fg_prob_post_target
diff_51_plus = avg_fg_prob_51_plus_15plus - avg_fg_prob_51_plus
print(f"Difference in Average FG Probability from 0 to {int(fg_target)} yards: {diff_pre_target:.0f}")
print(f"Difference in Average FG Probability from {int(fg_target)} to 50 yards: {diff_post_target:.0f}")
print(f"Difference in Average FG Probability from 51+ yards: {diff_51_plus:.0f}")


### ---- Final Visualization for Game Scenario #2 ---- ###
# Plot setup for the Football Field
fig, ax = plt.subplots(figsize=(24, 12))

# Set full field limits to depict what we need
ax.set_xlim(-10, 55)
ax.set_ylim(0, 10.01)
ax.set_yticks([])

# Shade Recommendation Zones based on FG Target Distance
ax.axvspan(0, fg_target - 17, color='darkgreen', alpha=0.8, label='High Confidence Zone')
ax.axvspan(fg_target - 17, 33, color='yellow', alpha=0.8, label='Caution Zone')
ax.axvspan(33, 45, color='red', alpha=0.8, label='Avoid Zone')

# # Add Labels for Each Zone at the Bottom of the Field
ax.text((fg_target - 17) / 2, 5, 'High Confidence Zone\n(Max 43-Yard FG)\n\n\n Avg. Zone\n FG Prob: {:.0f}%'.format(avg_fg_prob_pre_target), color='black', fontsize=16, fontweight='bold', ha='center', va='center')
ax.text((fg_target - 17 + 33) / 2, 5, 'Caution Zone\n(Max 50-Yard FG)\n\n\n Avg. Zone\nFG Prob: {:.0f}%'.format(avg_fg_prob_post_target), color='black', fontsize=16, fontweight='bold', ha='center', va='center')
ax.text((33 + 45) / 2, 5, 'Avoid Zone\n\n\n Avg. Zone\nFG Prob: {:.0f}%'.format(avg_fg_prob_51_plus), color='black', fontsize=16, fontweight='bold', ha='center', va='center')

# Black Rectangle to Separate Each Zone
ranges = [
    (0, fg_target - 17),
    (fg_target - 17, 33),
    (33, 45)
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

# Yard lines every 5 yards from 0 to 55
for x in range(0, 56, 5):
    ax.vlines(x, ymin=0, ymax=10, color='black', linewidth=1, linestyles='--')

# Yard line labels every 10 yards
yard_labels = list(range(10, 56, 10)) 
yard_line_positions = list(range(10, 56, 10))
ax.set_xticks(yard_line_positions, minor=False)
ax.set_xticklabels(yard_labels, minor=False, fontsize=12)

# # Yard Lines Labels for the Zones
minor_ticks = [0, fg_target - 17, 33, 45]
minor_labels = ['0', f'{int(fg_target - 17)}', '33', '45']
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(minor_labels, minor=True, fontsize=12, fontweight='bold', color='red')

# Line of Scrimmage X Label
ax.set_xlabel('Yardline (Line of Scrimmage)', fontsize=14, fontweight='bold')

# Add "CHARGERS" in both endzones
ax.text(-5, 5, 'C H A R G E R S', color='white', fontsize=64, fontweight='bold',
        ha='center', va='center', rotation=90)

# Add Chargers logo at midfield
logo_img = plt.imread('chargers_logo.webp')
ax.imshow(logo_img, extent=(45, 55, 3, 7), aspect='auto', alpha=0.8)

# Title
ax.set_title('Game Scenario #2: Free Agent Kicker FG Target Distance', fontweight='bold', fontsize=24)

# Add Description Text Box with Explanation at the Bottom (Bullet Points except first line)
description_text = (
    "**The Decision-Making Zones for the Week 7 Free Agent Kicker based on historical FG success probabilities for kickers attempting less than 15 kicks in a season**\n\n"
    "• 'High Confidence Zone' (0 - {fg_target_minus} Yardline): FGs are generally more successful (Within {diff_pre_target:.0f}% of League Avg.), with a maximum recommended distance of 43 yards for high confidence ({fg_prob_target:.0f}%).\n\n"
    "• 'Caution Zone' ({fg_target_minus1} - 33 Yardline): FGs success rates drop ({diff_post_target:.0f}% Below League Avg.), with a maximum cautious distance of 50 yards ({fg_prob_50:.0f}%). Lower Confidence when attempting FGs in this range.\n\n"
    "• 'Avoid Zone' (34 - 45 Yardline): FGs success rates are low, and attempts are discouraged."
).format(fg_target_minus=int(fg_target - 17), fg_prob_target=fg_prob_target, fg_prob_50=fg_prob_50, diff_post_target=diff_post_target, diff_pre_target=diff_pre_target, fg_target_minus1=int(fg_target - 17 + 1))

# Add Background Rectangle for Description
bbox_props = dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="lightgray")
ax.text(22.5, -1.2, description_text, fontsize=13, ha='center', va='top', bbox=bbox_props)

# Add a Black Line That Separates the Description from the Field on Figure
fig.add_artist(plt.Line2D([0.125, 0.875], [0.155, 0.155], color='black', linewidth=2, transform=fig.transFigure))
fig.add_artist(plt.Line2D([0.125, 0.125], [0, 0.155], color='black', linewidth=2, transform=fig.transFigure))
fig.add_artist(plt.Line2D([0.875, 0.875], [0, 0.155], color='black', linewidth=2, transform=fig.transFigure))

# Layout fix
plt.tight_layout()
plt.savefig('Andrew Purvis - Game Scenario 2 (Free Agent Kicker FG Target Line).png', dpi=300)


### ---- Additional Visualization for Context ---- ###
# Plot FG% by Distance
plt.figure(figsize=(12, 6))
plt.plot(fg_stats['kick_distance'], fg_stats['fg_percentage'], marker='o', label='15+ Attempts')
plt.plot(fg_stats2['kick_distance'], fg_stats2['fg_percentage'], marker='o', label='15- Attempts')
plt.title('Field Goal Percentage by Distance')
plt.xlabel('Distance (yards)')
plt.ylabel('Field Goal Percentage')
plt.grid()
plt.legend()
plt.savefig('Scenario 2 Context Images/Weighted FG Percentage by Distance for Scenario 2 (Kicker Attempts).png')
plt.show()