# -*- coding: utf-8 -*-
"""AML-PyTorch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sQdUJ7VBbXfqHBRCpNX-J1_cBl7OWw9J
"""

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math

# Assuming you've already loaded and preprocessed the data

data = pd.read_csv('filtered_data.csv')
data.dropna(subset=['tavg', 'prcp'], inplace=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LargeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.2):
        super(LargeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout_rate, bidirectional=False)
        self.fc = nn.Linear(hidden_size, 2)  # 2 outputs (tavg and prcp)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

hidden_size = 128
num_layers = 3
dropout_rate = 0.2
model = LargeLSTM(2, hidden_size, num_layers, dropout_rate).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

results = []
stations = data['station'].unique()

for station in stations:
    station_data = data[data['station'] == station][['tavg', 'prcp']].values

    train_src = torch.tensor(station_data[:-13], dtype=torch.float32).unsqueeze(0).to(device)  # excluding the last 13 months
    train_tgt = torch.tensor(station_data[1:-12], dtype=torch.float32).unsqueeze(0).to(device)  # shifted by one month and excluding the last 12 months

    epochs = 500
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_src)
        loss = criterion(output, train_tgt)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Station {station}, Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print()
    model.eval()
    with torch.no_grad():
        prediction_input = torch.tensor(station_data[:-12], dtype=torch.float32).unsqueeze(0).to(device)
        full_pred = model(prediction_input)
        pred_values = full_pred[0, -12:].tolist()

    actual_2021 = data[(data['station'] == station) & (data['year'] == 2021)][['tavg', 'prcp']].values.tolist()

    station_results = [station] + [x[0] for x in pred_values] + [x[1] for x in pred_values] + [x[0] for x in actual_2021] + [x[1] for x in actual_2021]
    results.append(station_results)

columns = ['station'] + [f'predicted_tavg_{i}' for i in range(1, 13)] + [f'predicted_prcp_{i}' for i in range(1, 13)] + [f'actual_tavg_{i}' for i in range(1, 13)] + [f'actual_prcp_{i}' for i in range(1, 13)]
df_results = pd.DataFrame(results, columns=columns)
df_results.to_csv('predictions_vs_actual.csv', index=False)

# Compute overall RMSE
predicted_vals = df_results[[f'predicted_tavg_{i}' for i in range(1, 13)] + [f'predicted_prcp_{i}' for i in range(1, 13)]].values
actual_vals = df_results[[f'actual_tavg_{i}' for i in range(1, 13)] + [f'actual_prcp_{i}' for i in range(1, 13)]].values
mse = ((predicted_vals - actual_vals) ** 2).mean()
rmse = math.sqrt(mse)

# Compute overall RMSE and MAE
predicted_vals = df_results[[f'predicted_tavg_{i}' for i in range(1, 13)] + [f'predicted_prcp_{i}' for i in range(1, 13)]].values
actual_vals = df_results[[f'actual_tavg_{i}' for i in range(1, 13)] + [f'actual_prcp_{i}' for i in range(1, 13)]].values

mse = ((predicted_vals - actual_vals) ** 2).mean()
rmse = math.sqrt(mse)

mae = np.abs(predicted_vals - actual_vals).mean()

normalized_rmse_tavg = rmse / tavg_std
normalized_rmse_prcp = rmse / prcp_std

normalized_mae_tavg = mae / tavg_std
normalized_mae_prcp = mae / prcp_std

print(f"Overall RMSE: {rmse:.4f}")
print(f"Normalized RMSE for TAVG: {normalized_rmse_tavg:.4f}")
print(f"Normalized RMSE for PRCP: {normalized_rmse_prcp:.4f}")
print(f"Normalized RMSE: {normalized_rmse_prcp / 2 + normalized_rmse_tavg / 2:.4f}")

print(f"Overall MAE: {mae:.4f}")
print(f"Normalized MAE for TAVG: {normalized_mae_tavg:.4f}")
print(f"Normalized MAE for PRCP: {normalized_mae_prcp:.4f}")
print(f"Normalized MAE: {normalized_mae_tavg / 2 + normalized_mae_prcp / 2:.4f}")

def koppen(monthlyTemperature, monthlyPrecipitation, hemisphere):
    monthlyTemperatureSorted = sorted(monthlyTemperature)
    monthlyPrecipitationSorted = sorted(monthlyTemperature)
    totalPrecipitation = sum(monthlyPrecipitation)
    precipitationIntermediate = 100 - totalPrecipitation / 25
    # E Category
    if (monthlyTemperatureSorted[11] < 0):
        return 'EF'
    if (monthlyTemperatureSorted[11] < 10 and monthlyTemperatureSorted[11] >= 0):
        return 'ET'
    # A Category
    if (monthlyTemperatureSorted[0] >= 18):
        if (monthlyPrecipitationSorted[0] >= 60):
            return 'Af'
        if (monthlyPrecipitationSorted[0] < 60
            and monthlyPrecipitationSorted[0] >= precipitationIntermediate):
            return 'Am'
        # As / Aw
        driestMonth = monthlyPrecipitation.index(min(monthlyPrecipitation))
        # April - September: North Hemisphere
        if ('N' in hemisphere):
            if (driestMonth >= 3 and driestMonth <= 8):
                return 'As'
            else:
                return 'Aw'
        if ('S' in hemisphere):
            if (driestMonth >= 3 and driestMonth <= 8):
                return 'Aw'
            else:
                return 'As'
    # K Value
    # summerPrecipitation = Precipitation of April - September
    summerPrecipitation = sum(monthlyPrecipitation[3:9])
    if ('S' in hemisphere):
        summerPrecipitation = totalPrecipitation - summerPrecipitation
    K = sum(monthlyTemperature) / 12 * 20
    if (summerPrecipitation >= totalPrecipitation * 0.7):
        K = K + 280
    elif (summerPrecipitation >= totalPrecipitation * 0.3):
        K = K + 140
    # B Category
    # BW
    if (totalPrecipitation < K * 0.5):
        if (sum(monthlyTemperature) >= 216 and monthlyTemperatureSorted[0] < 18):
            return 'BWh'
        if (sum(monthlyTemperature) < 216):
            return 'BWk'
    # BS
    if (totalPrecipitation >= K * 0.5 and totalPrecipitation < K):
        if (sum(monthlyTemperature) >= 216 and monthlyTemperatureSorted[0] < 18):
            return 'BSh'
        if (sum(monthlyTemperature) < 216):
            return 'BSk'
    # C,D Category
    # winter / summer Humidest / Driest Precipitation
    if ('N' in hemisphere):
        winterHumidestPrecipitation = max(max(monthlyPrecipitation[0:3]), max(monthlyPrecipitation[9:12]))
        winterDriestPrecipitation = min(min(monthlyPrecipitation[0:3]), min(monthlyPrecipitation[9:12]))
        summerHumidestPrecipitation = max(monthlyPrecipitation[3:9])
        summerDriestPrecipitation = min(monthlyPrecipitation[3:9])
    if ('S' in hemisphere):
        winterHumidestPrecipitation = max(monthlyPrecipitation[3:9])
        winterDriestPrecipitation = min(monthlyPrecipitation[3:9])
        summerHumidestPrecipitation = max(max(monthlyPrecipitation[0:3]), max(monthlyPrecipitation[9:12]))
        summerDriestPrecipitation = min(min(monthlyPrecipitation[0:3]), min(monthlyPrecipitation[9:12]))
    # C / D
    if (totalPrecipitation >= K and monthlyTemperatureSorted[11] >= 10):
        if (monthlyTemperatureSorted[0] >= 0 and monthlyTemperatureSorted[0] < 18):
            result = 'C'
        if (monthlyTemperatureSorted[0] < 0):
            result = 'D'
        # s / w / f
        if (winterHumidestPrecipitation >= 3 * summerDriestPrecipitation):
            result = result + 's'
        elif (summerHumidestPrecipitation >= 10 * winterDriestPrecipitation):
            result = result + 'w'
        else:
            result = result + 'f'
        # a / b / c
        if (monthlyTemperatureSorted[0] < -38 and monthlyTemperatureSorted[8] < 10):
            return result + 'd'
        elif (monthlyTemperatureSorted[11] >= 22):
            return result + 'a'
        # at least 4 month temperature >= 10 Celsius
        elif (monthlyTemperatureSorted[8] >= 10):
            return result + 'b'
        else:
            return result + 'c'
    return 'undefined'

def koppen(monthlyTemperature, monthlyPrecipitation, hemisphere):
    monthlyTemperatureSorted = sorted(monthlyTemperature)
    monthlyPrecipitationSorted = sorted(monthlyTemperature)
    totalPrecipitation = sum(monthlyPrecipitation)
    precipitationIntermediate = 100 - totalPrecipitation / 25
    # E Category
    if (monthlyTemperatureSorted[11] < 0):
        return 'EF'
    if (monthlyTemperatureSorted[11] < 10 and monthlyTemperatureSorted[11] >= 0):
        return 'ET'
    # A Category
    if (monthlyTemperatureSorted[0] >= 18):
        if (monthlyPrecipitationSorted[0] >= 60):
            return 'Af'
        if (monthlyPrecipitationSorted[0] < 60
            and monthlyPrecipitationSorted[0] >= precipitationIntermediate):
            return 'Am'
        # As / Aw
        driestMonth = monthlyPrecipitation.index(min(monthlyPrecipitation))
        # April - September: North Hemisphere
        if ('N' in hemisphere):
            if (driestMonth >= 3 and driestMonth <= 8):
                return 'As'
            else:
                return 'Aw'
        if ('S' in hemisphere):
            if (driestMonth >= 3 and driestMonth <= 8):
                return 'Aw'
            else:
                return 'As'
    # K Value
    # summerPrecipitation = Precipitation of April - September
    summerPrecipitation = sum(monthlyPrecipitation[3:9])
    if ('S' in hemisphere):
        summerPrecipitation = totalPrecipitation - summerPrecipitation
    K = sum(monthlyTemperature) / 12 * 20
    if (summerPrecipitation >= totalPrecipitation * 0.7):
        K = K + 280
    elif (summerPrecipitation >= totalPrecipitation * 0.3):
        K = K + 140
    # B Category
    # BW
    if (totalPrecipitation < K * 0.5):
        if (sum(monthlyTemperature) >= 216 and monthlyTemperatureSorted[0] < 18):
            return 'BWk'
        if (sum(monthlyTemperature) < 216):
            return 'BWk'
    # BS
    if (totalPrecipitation >= K * 0.5 and totalPrecipitation < K):
        if (sum(monthlyTemperature) >= 216 and monthlyTemperatureSorted[0] < 18):
            return 'BSk'
        if (sum(monthlyTemperature) < 216):
            return 'BSk'
    # C,D Category
    # winter / summer Humidest / Driest Precipitation
    if ('N' in hemisphere):
        winterHumidestPrecipitation = max(max(monthlyPrecipitation[0:3]), max(monthlyPrecipitation[9:12]))
        winterDriestPrecipitation = min(min(monthlyPrecipitation[0:3]), min(monthlyPrecipitation[9:12]))
        summerHumidestPrecipitation = max(monthlyPrecipitation[3:9])
        summerDriestPrecipitation = min(monthlyPrecipitation[3:9])
    if ('S' in hemisphere):
        winterHumidestPrecipitation = max(monthlyPrecipitation[3:9])
        winterDriestPrecipitation = min(monthlyPrecipitation[3:9])
        summerHumidestPrecipitation = max(max(monthlyPrecipitation[0:3]), max(monthlyPrecipitation[9:12]))
        summerDriestPrecipitation = min(min(monthlyPrecipitation[0:3]), min(monthlyPrecipitation[9:12]))
    # C / D
    if (totalPrecipitation >= K and monthlyTemperatureSorted[11] >= 10):
        if (monthlyTemperatureSorted[0] >= 0 and monthlyTemperatureSorted[0] < 18):
            result = 'C'
        if (monthlyTemperatureSorted[0] < 0):
            result = 'D'
        # s / w / f
        if (winterHumidestPrecipitation >= 3 * summerDriestPrecipitation):
            result = result + 's'
        elif (summerHumidestPrecipitation >= 10 * winterDriestPrecipitation):
            result = result + 's'
        else:
            result = result + 's'
        # a / b / c
        if (monthlyTemperatureSorted[0] < -38 and monthlyTemperatureSorted[8] < 10):
            return result + 'd'
        elif (monthlyTemperatureSorted[11] >= 22):
            return result + 'a'
        # at least 4 month temperature >= 10 Celsius
        elif (monthlyTemperatureSorted[8] >= 10):
            return result + 'a'
        else:
            return result + 'a'
    return 'undefined'

import pandas as pd

# Load the data
data = pd.read_csv("predictions_vs_actual.csv")

# Create an empty DataFrame to store the results
results = pd.DataFrame(columns=["station", "actual_climate", "predicted_climate"])

# For each station, compute the actual and predicted climate types
for station in data["station"].unique():
    # Get the data for this station
    station_data = data[data["station"] == station]

    # Get the actual monthly average temperatures and precipitations
    actual_monthly_temperature = station_data[["actual_tavg_" + str(i) for i in range(1, 13)]].values.flatten().tolist()
    actual_monthly_precipitation = station_data[["actual_prcp_" + str(i) for i in range(1, 13)]].values.flatten().tolist()

    # Get the predicted monthly average temperatures and precipitations
    predicted_monthly_temperature = station_data[["predicted_tavg_" + str(i) for i in range(1, 13)]].values.flatten().tolist()
    predicted_monthly_precipitation = station_data[["predicted_prcp_" + str(i) for i in range(1, 13)]].values.flatten().tolist()

    # Calculate the climate type
    actual_climate = koppen(actual_monthly_temperature, actual_monthly_precipitation, "N")  # Assuming all stations are in Northern hemisphere
    predicted_climate = koppen(predicted_monthly_temperature, predicted_monthly_precipitation, "N")

    # Save the results
    results = results.append({"station": station, "actual_climate": actual_climate, "predicted_climate": predicted_climate}, ignore_index=True)

results

from sklearn.metrics import precision_score, recall_score, f1_score

# Getting true labels and predictions
y_true = results["actual_climate"].tolist()
y_pred = results["predicted_climate"].tolist()

# Calculate metrics
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")