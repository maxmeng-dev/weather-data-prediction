import requests
import gzip
import csv
import json
import os
import concurrent.futures
import threading

# Open JSON file and load into data
with open('stations.json') as f:
    data = json.load(f)

stations = list()

for d in data:
    stations.append(d['id'])

path_hourly = './dataset/hourly'
path_daily = './dataset/daily'
path_monthly = './dataset/monthly'
path_normals = './dataset/normals'

# Download monthly data for each station


def download_station(station):
    print("thread number: ", threading.active_count(), end="\r")

    def download(req_url, folder_path, station_id):
        response = requests.get(req_url)

        if response.status_code != 200:
            return

        filename = os.path.basename(req_url)
        path = os.path.join(os.getcwd(), folder_path, filename)
        with open(path, 'wb') as f:
            f.write(response.content)

        # Decompress data
        with gzip.open(path, 'rb') as f_in:
            with open(os.path.join(folder_path, station_id) + '.csv', 'wb') as f_out:
                f_out.write(f_in.read())
        
        os.remove(path)

     # Build URL for this station
    url_h = f'https://bulk.meteostat.net/v2/hourly/{station}.csv.gz'
    url_d = f'https://bulk.meteostat.net/v2/daily/{station}.csv.gz'
    url_m = f'https://bulk.meteostat.net/v2/monthly/{station}.csv.gz'
    url_n = f"https://bulk.meteostat.net/v2/normals/{station}.csv.gz"
    #download(url_h, path_hourly, station)
    #download(url_d, path_daily, station)
    download(url_m, path_monthly, station)
    #download(url_n, path_normals, station)


# Download all stations using threads
with concurrent.futures.ThreadPoolExecutor(max_workers=1500) as executor:
    # os.makedirs(path_daily, exist_ok=True)
    os.makedirs(path_monthly, exist_ok=True)
    # os.makedirs(path_hourly, exist_ok=True)
    # os.makedirs(path_normals, exist_ok=True)
    results = executor.map(download_station, stations)

print('Download complete                     ')
