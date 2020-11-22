import pathlib
import requests

from geopy.distance import distance as coord_distance
import pandas as pd



# SAVE_FOLDER_PATH = 'C:/Users/Arenter/Documents/Python Scripts/rent/Avito/data/raw'
PROJECT_FOLDER_PATH = pathlib.Path().resolve().parents[1]
SAVE_FOLDER_PATH = PROJECT_FOLDER_PATH / 'data/raw'
KM_ZERO = {'lat': 55.755919, 'lng': 37.617589}

response = requests.get('https://api.hh.ru/metro/1')
response_json = response.json()
moscow = response_json['lines']

stations_coord = {}
for line_iter in range(len(moscow)):
    for station_iter in range(len(moscow[line_iter]['stations'])):
        station_dict = moscow[line_iter]['stations'][station_iter]
        station_name = station_dict['name'].strip()
        stations_coord[station_name] = {'lat': station_dict['lat'], 
                                        'lng': station_dict['lng'],
                                        'line': moscow[line_iter]['name']}
stations_coord['Лефортово'] = {'lat': 55.764444,
                               'lng': 37.702777,
                               'line': 'Некрасовская'}

station_names = stations_coord.keys()
for name in station_names:
    center_distance = coord_distance((stations_coord[name]['lat'], stations_coord[name]['lng']),
                                     (KM_ZERO['lat'], KM_ZERO['lng']))
    stations_coord[name]['center_distance'] = center_distance.km

stations_df = pd.DataFrame.from_dict(stations_coord, orient='index')
stations_df.loc[["Белорусская", "Курская"], 'line'] = "Кольцевая"
stations_df['mcc'] = (stations_df['line']=="МЦК").astype(int)
stations_df['circle'] = (stations_df['line']=="Кольцевая").astype(int)
stations_df.sort_index(inplace=True)
stations_df[stations_df['line'] == 'Некрасовская']
stations_df.to_csv(SAVE_FOLDER_PATH / 'stations_df2.csv')