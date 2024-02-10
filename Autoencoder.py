#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

# Load the CSV file
data = pd.read_csv('/Users/mxx/Desktop/Final/concat-filter.csv')

# Drop unnecessary columns
data = data[['station', 'year', 'month', 'tavg', 'prcp']]

# Create a mask for the desired date range
desired_dates = [(year, month) for year in range(1977, 2022) for month in range(1, 13)]

# Filter stations
valid_stations = []

for station in data['station'].unique():
    station_data = data[data['station'] == station]
    valid = True
    
    for year, month in desired_dates:
        monthly_data = station_data[(station_data['year'] == year) & (station_data['month'] == month)]
        
        # Check if there's a record for this month and if tavg and prcp are non-null
        if len(monthly_data) == 0 or monthly_data['tavg'].isnull().values[0] or monthly_data['prcp'].isnull().values[0]:
            valid = False
            break

    if valid:
        valid_stations.append(station)

# Filter the data to include only valid stations
filtered_data = data[data['station'].isin(valid_stations)]

# Save the filtered data
filtered_data.to_csv('filtered_data.csv', index=False)


# In[9]:


len(valid_stations)


# In[26]:


import csv

station_data = {}

with open('/Users/mxx/Desktop/Final/filtered_data.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        station = row['station']
        year = int(row['year'])
        if (year >= 1977 and year <= 2021):
            month = int(row['month'])
            prcp = row['prcp']

            if station not in station_data:
                station_data[station] = {'station': station}

            key = f'year={year},month={month}'
            station_data[station][key] = prcp

with open('/Users/mxx/Desktop/Final/filtered_prcp.csv', 'w', newline='') as csvfile:
    fieldnames = ['station']
    for year in range(1977, 2022):
        for month in range(1, 13):
            fieldnames.append(f'year={year},month={month}')

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for station in station_data.values():
        writer.writerow(station)


# In[27]:


import csv

station_data = {}

with open('/Users/mxx/Desktop/Final/filtered_data.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        station = row['station']
        year = int(row['year'])
        if (year >= 1977 and year <= 2021):
            month = int(row['month'])
            tavg = row['tavg']

            if station not in station_data:
                station_data[station] = {'station': station}

            key = f'year={year},month={month}'
            station_data[station][key] = tavg

with open('/Users/mxx/Desktop/Final/filtered_tavg.csv', 'w', newline='') as csvfile:
    fieldnames = ['station']
    for year in range(1977, 2022):
        for month in range(1, 13):
            fieldnames.append(f'year={year},month={month}')

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for station in station_data.values():
        writer.writerow(station)


# In[28]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load the data
tavg_df = pd.read_csv("/Users/mxx/Desktop/Final/filtered_tavg.csv", index_col=0)
prcp_df = pd.read_csv("/Users/mxx/Desktop/Final/filtered_prcp.csv", index_col=0)

# Merge the data
data = pd.concat([tavg_df, prcp_df], axis=1)

# Define the Autoencoder model using PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Train the Autoencoder
model = Autoencoder(data.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
data_tensor = torch.tensor(data.values, dtype=torch.float32)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(data_tensor)
    loss = criterion(outputs, data_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Obtain the encoding for each station
encoded_data = model.encoder(data_tensor)

# Compute the differences in the encoding for each station
differences = np.sum(np.abs(encoded_data.detach().numpy()), axis=1)

# Find the top five stations with the most change
sorted_indices = np.argsort(differences)[::-1]
top_five_stations = tavg_df.index[sorted_indices[:5]]

print("The top five stations with the most change are:")
for station in top_five_stations:
    print(station)


# Zugspitze (Germany) ID=10961
# Miami (USA) ID=72202
# Jacksonville (USA) ID=72206
