import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
data = pd.read_csv('dataset/concat.csv')

# Randomly select 50 data points
random_data = data.sample(n=50)

# Create a pivot table for the heatmap
pivot_data = data.pivot_table(index='station', columns=['year', 'month'], values='tavg', aggfunc='count', fill_value=0)
total_months = pivot_data.columns.levels[1].max()
coverage_data = pivot_data / total_months

# Set the plotting style
sns.set(style="whitegrid")

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_data, cmap='YlGnBu', annot=True, fmt='g')
plt.title('Weather Data Heatmap (Random 50 Stations)')
plt.xlabel('Year')
plt.ylabel('Station ID')

# Save the image to a file
plt.savefig('heatmap_random.png')
