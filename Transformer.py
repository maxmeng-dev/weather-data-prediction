import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
print("t",torch.__version__)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math

# Transformer Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.linear = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads,
                                          num_encoder_layers=num_layers,
                                          dropout=dropout)
        self.fc = nn.Linear(model_dim, input_dim)

    def forward(self, src, tgt=None):
        src = self.linear(src)
        if tgt is None:
            tgt = src
        else:
            tgt = self.linear(tgt)
        output = self.transformer(src, tgt)
        return self.fc(output)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math

# Load data
data = pd.read_csv('filtered_data.csv')

# Preprocess data
data['tavg'] = pd.to_numeric(data['tavg'], errors='coerce')
data['prcp'] = pd.to_numeric(data['prcp'], errors='coerce')
data.dropna(subset=['tavg', 'prcp'], inplace=True)

# Normalize data
tavg_mean, tavg_std = data['tavg'].mean(), data['tavg'].std()
prcp_mean, prcp_std = data['prcp'].mean(), data['prcp'].std()
data['tavg_norm'] = (data['tavg'] - tavg_mean) / tavg_std
data['prcp_norm'] = (data['prcp'] - prcp_mean) / prcp_std


results_normalised = []
results=[]
stations = data['station'].unique()

for station in stations:
    train_mask = (data['station'] == station) & (data['year'].between(1977, 2019))
    val_mask = (data['station'] == station) & (data['year'] == 2020)
    train_data = data[train_mask][['tavg_norm', 'prcp_norm']].values
    val_data = data[val_mask][['tavg_norm', 'prcp_norm']].values

    train_src = torch.tensor(train_data, dtype=torch.float32).unsqueeze(0).to(device)
    # Decoder target should have the same length as train_src but ending in val_data (2020 data)
    train_tgt = np.vstack([train_data[:-12], val_data])
    train_tgt = torch.tensor(train_tgt, dtype=torch.float32).unsqueeze(0).to(device)

    model = TimeSeriesTransformer(2, 512, 8, 3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_src, train_tgt)  # Encoder uses train_data, Decoder tries to predict val_data
        loss = criterion(output[:, -12:], train_tgt[:, -12:])  # Focus loss on the last 12 months (2020 data)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Station {station}, Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        # For prediction, provide only the encoder input. Let the decoder generate predictions.
        pred_2021 = model(train_src)
        pred_2021 = pred_2021[:, -12:]

    # Retrieve last 12 months for predictions
    pred_values = pred_2021[0, -12:].tolist()

    actual_tavg_norm_2021 = data[(data['station'] == station) & (data['year'] == 2021)][['tavg_norm']].values.flatten().tolist()
    actual_prcp_norm_2021 = data[(data['station'] == station) & (data['year'] == 2021)][['prcp_norm']].values.flatten().tolist()

    pred_tavg_norm = [x[0] for x in pred_values]
    pred_prcp_norm = [x[1] for x in pred_values]

    station_results = [station] + pred_tavg_norm + pred_prcp_norm + actual_tavg_norm_2021 + actual_prcp_norm_2021
    results_normalised.append(station_results)

    #### Inverse normalization
    pred_tavg = [x[0] * tavg_std + tavg_mean for x in pred_values]
    pred_prcp = [x[1] * prcp_std + prcp_mean for x in pred_values]
    actual_tavg_2021 = data[(data['station'] == station) & (data['year'] == 2021)][['tavg']].values.flatten().tolist()
    actual_prcp_2021 = data[(data['station'] == station) & (data['year'] == 2021)][['prcp']].values.flatten().tolist()

    station_results = [station] + pred_tavg + pred_prcp + actual_tavg_2021 + actual_prcp_2021
    results.append(station_results)



# Create final CSV with origin data
columns = ['station'] + [f'predicted_tavg_{i}' for i in range(1, 13)] + [f'predicted_prcp_{i}' for i in range(1, 13)] + [f'actual_tavg_{i}' for i in range(1, 13)] + [f'actual_prcp_{i}' for i in range(1, 13)]
df_results = pd.DataFrame(results, columns=columns)
df_results.to_csv('predictions_vs_actual.csv', index=False)

# Compute overall RMSE using norm data
columns = ['station'] + [f'predicted_tavg_{i}' for i in range(1, 13)] + [f'predicted_prcp_{i}' for i in range(1, 13)] + [f'actual_tavg_{i}' for i in range(1, 13)] + [f'actual_prcp_{i}' for i in range(1, 13)]
df_results_normalised = pd.DataFrame(results_normalised, columns=columns)
df_results_normalised.to_csv('predictions_vs_actual_norm.csv', index=False)

predicted_vals = df_results_normalised[[f'predicted_tavg_{i}' for i in range(1, 13)] + [f'predicted_prcp_{i}' for i in range(1, 13)]].values
actual_vals = df_results_normalised[[f'actual_tavg_{i}' for i in range(1, 13)] + [f'actual_prcp_{i}' for i in range(1, 13)]].values

mse = ((predicted_vals - actual_vals) ** 2).mean()
rmse = math.sqrt(mse)

print(f"Overall RMSE: {rmse:.4f}")

mae = abs(predicted_vals - actual_vals).mean()
print(f"Overall MAE: {mae:.4f}")

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
