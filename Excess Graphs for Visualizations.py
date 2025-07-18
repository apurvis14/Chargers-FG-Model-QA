import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

### ---- Load All Data ---- ###
data_all = pd.read_csv('data/2016-2024 PBP Data.csv')
data_features = pd.read_csv('data/FG Data1.csv')
data_predictions = pd.read_csv('data/Full FG Data with Predictions.csv')

### ---- Excess Graphs for Visualizations ---- ###
# Create a Scatter Plot with EP vs Yardline_100 on 4th Down
data_4th_down = data_all[data_all['down'] == 4]

# Create Avg EP by Yardline_100
avg_ep_by_yardline = data_4th_down.groupby('yardline_100')['ep'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=avg_ep_by_yardline['yardline_100'], y=avg_ep_by_yardline['ep'], alpha=0.3)
plt.title('Expected Points (EP) vs Yardline 100 on 4th Down')
plt.xlabel('Yardline 100')
plt.ylabel('Expected Points (EP)')
plt.grid()
plt.savefig('Excess Visualizations/EP vs Yardline 100 on 4th Down Scatterplot.png', dpi=300)
plt.show()


# Create A Histogram of Wind Speed for Scenario 1
plt.figure(figsize=(10, 6))
bins = np.arange(1, data_features['wind'].max() + 2, 1)
sns.histplot(data_features['wind'].dropna(), bins=bins, kde=False)
plt.title('Histogram of Wind Speed')
plt.xlabel('Wind Speed (mph)')
plt.ylabel('Frequency')
plt.grid()
plt.savefig('Excess Visualizations/Histogram of Wind Speed.png', dpi=300)
plt.show()

# # FG Prob vs EP on Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_features['fg_prob'], y=data_features['ep'], alpha=0.5)
plt.title('FG Probability vs Expected Points (EP)')
plt.xlabel('FG Probability')
plt.ylabel('Expected Points (EP)')
plt.grid()
plt.savefig('Excess Visualizations/FG Probability vs Expected Points Scatterplot.png', dpi=300)
plt.show()

# Create a Scatterplot of Distance vs League FG% at that Distance
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_features['kick_distance'], y=data_features['fg_pct_by_distance_overall'], alpha=0.5)
plt.title('Kick Distance vs League FG% at that Distance')
plt.xlabel('Kick Distance')
plt.ylabel('League FG% at that Distance')
plt.ylim(0, 1)
plt.grid()
plt.savefig('Excess Visualizations/Kick Distance vs League FG% at that Distance Scatterplot.png', dpi=300)
plt.show()

### ---- Load Model Data ---- ###
data_model = pd.read_csv('data/FG Predictions.csv') 

# Plot Ensemble Histogram
plt.figure(figsize=(8, 6))
plt.hist(data_model['ensemble_prob'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of Ensemble Predicted Probabilities')
plt.xlabel('Predicted Probability of Success')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('Excess Visualizations/Distribution of Ensemble Predicted Probabilities.png')
plt.show()

# Plot Logistic Histogram
plt.figure(figsize=(8, 6))
plt.hist(data_model['logistic_prob'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of Logistic Regression Predicted Probabilities')
plt.xlabel('Predicted Probability of Success')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('Excess Visualizations/Distribution of Logistic Regression Predicted Probabilities.png')
plt.show()

# Plot XGB Histogram
plt.figure(figsize=(8, 6))
plt.hist(data_model['xgb_prob'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of XGBoost Predicted Probabilities')
plt.xlabel('Predicted Probability of Success')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('Excess Visualizations/Distribution of XGBoost Predicted Probabilities.png')
plt.show()

# Plot Logistic vs FG Probability from NFL Verse
plt.figure(figsize=(10, 6))
plt.scatter(data_model['fg_prob'], data_model['logistic_prob'], alpha=0.5)
plt.title('Logistic Predicted Probability vs FG Probability')
plt.xlabel('FG Probability')
plt.ylabel('Logistic Predicted Probability')
plt.grid(True)
plt.savefig('Excess Visualizations/Logistic Predicted Probability vs FG Probability.png')
plt.show()

# Plot XGB vs FG Probability from NFL Verse
plt.figure(figsize=(10, 6))
plt.scatter(data_model['fg_prob'], data_model['xgb_prob'], alpha=0.5)
plt.title('XGBoost Predicted Probability vs FG Probability')
plt.xlabel('FG Probability')
plt.ylabel('XGBoost Predicted Probability')
plt.grid(True)
plt.savefig('Excess Visualizations/XGBoost Predicted Probability vs FG Probability.png')
plt.show()