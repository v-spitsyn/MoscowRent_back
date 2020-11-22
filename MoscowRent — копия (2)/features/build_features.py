from datetime import datetime, timedelta
import pathlib
import pickle
from typing import List, Tuple, Optional

import Levenshtein
import numpy as np
import pandas as pd



PROJECT_FOLDER_PATH = pathlib.Path().resolve().parents[1]
FLATS_FOLDER_PATH = PROJECT_FOLDER_PATH / 'data/raw'
STATIONS_DF_PATH = PROJECT_FOLDER_PATH / 'data/raw/stations_df.csv'
FEATURES_FROM_DATE = pd.to_datetime('2020-08-01').date()
FEATURES_TO_DATE = pd.to_datetime('2020-08-14').date()
FEATURES_FILE = 'features_{}_{}.csv'.format(FEATURES_FROM_DATE, FEATURES_TO_DATE)
FEATURES_SAVE_PATH = PROJECT_FOLDER_PATH / f'data/processed/{FEATURES_FILE}'



def pickle_to_df(file_path: str) -> pd.DataFrame:
    with open(file_path, 'rb') as f:
        flats = pickle.load(f)
    df = pd.DataFrame(flats)
    return df

def bind_df() -> pd.DataFrame:
    flats_paths = FLATS_FOLDER_PATH.glob('flats_*.pickle')
    dfs_list = [pickle_to_df(path) for path in flats_paths]
    return pd.concat(dfs_list, ignore_index=True)

def parse_distance(distance: Optional[str]) -> Optional[float]:
    if distance is None:
        return None
    try:
        
        distance_dotted = distance.split()[0].replace(',', '.')  
        units = distance.split()[1]
        if units == "км":
            distance_hundred_meters = float(distance_dotted)*10
        else:
            distance_hundred_meters = float(distance_dotted)/100
    except Exception:
        print(distance)
        return(None)
    return distance_hundred_meters

def parse_header(row: pd.DataFrame) -> List[int]:
    try:
        header_split = row['header'].split(',')
        raw_rooms = header_split[0]
        raw_area = header_split[1]
        raw_floors = header_split[2]
        studio = 1 if raw_rooms.split()[0] == 'Студия' else 0
        if studio:
            n_rooms = 1
        else:
            n_rooms = int(raw_rooms[0])
        area_split = raw_area.split()
        area = int(round(float(area_split[0])))
        floors_split = raw_floors.split()[0].split('/')
        floor = int(floors_split[0])
        n_floors = int(floors_split[1])
        header_fields = (studio, n_rooms, area, floor, n_floors)
    except Exception:
        print(row[['ref', 'header']])
        return None
    return header_fields

def parse_price(price: str) -> float:
    price_rub = float(price)
    price_thousands = price_rub/1000
    return price_thousands

def parse_publication_datetime(row: pd.DataFrame) -> Tuple[datetime.date, datetime.time]:
    try:
        published_split = row['published'].split()
        units_match = {'с': 'seconds',
                       'м': 'minutes',
                       'ч': 'hours',
                       'н': 'weeks',
                       'д': 'days'}
        unit = units_match[published_split[1][0]]
        if unit == 'seconds':
            publication_datetime = row['parsing_time']
        else:
            num_units = int(published_split[0])
            timedelta_kwargs = {unit: num_units}
            publication_datetime = row['parsing_time'] - timedelta(**timedelta_kwargs)
        if publication_datetime.date() == row['parsing_time'].date():
            publication_time = publication_datetime.time()
        else:
            publication_time = None
        publication_date = publication_datetime.date()
    except Exception:
        print(row[['ref', 'published', 'parsing__date']])
        return None
    return [publication_date, publication_time]

def remove_nonmoscow() -> None:
    df.dropna(subset=['station', 'distance', 'address'], inplace=True)
    df.drop(df[df['ref'].str.contains('zelenograd')].index, inplace=True)
    df.drop(df[df['address'].str.contains('Зеленоград')].index, inplace=True)

def remove_special_chars() -> None:
    str_cols = df.dtypes[df.dtypes=='object'].index
    for col in str_cols:
        for char in ('\n', '\t'):
            df[col] = (df[col].str.replace(char, '')
                              .str.strip())

def match_station(parsed_station: Optional[str]) -> Optional[str]:
    if parsed_station is None:
        return None
    vectorized_Levenshtein_distance = np.vectorize(Levenshtein.distance)
    string_distances = vectorized_Levenshtein_distance(parsed_station, stations_df.index)
    matched_station = stations_df.index[np.argmin(string_distances)]
    return matched_station

def cols_from_parsed() -> None:
    funcs_to_series = {parse_distance: ['distance', 'station_distance'],
                       match_station: ['station', 'matched_station'],
                       parse_price: ['price', 'rent']
                       } 
    funcs_to_df = {parse_publication_datetime: ['pub_date', 'pub_time'],
                   parse_header: ['studio', 'n_rooms', 'area', 'floor', 'n_floors']
                   }
    for fun in funcs_to_series:
        apply_col = funcs_to_series[fun][0]
        create_col = funcs_to_series[fun][1]
        df[create_col] = df[apply_col].apply(fun) 
    for fun in funcs_to_df:
        create_cols = funcs_to_df[fun]
        df[create_cols] = df.apply(fun, axis=1, result_type='expand')

def form_features(source_df: pd.DataFrame) -> pd.DataFrame:
    features = (source_df.sort_values(['ref', 'parsing_time'])
                         .groupby('ref')
                         .first())
    features.dropna(subset=['studio', 'n_rooms', 'area', 'floor', 'rent'], inplace=True)
    # features.sort_values('parsing_time', inplace=True)
    features = features[features['pub_date'] >= FEATURES_FROM_DATE]
    features['y'] = features['rent'] / features['area']
    drop_cols = ['address', 'commission', 'distance', 'header', 'matched_station', 
                 'n_floors','parsing_time','published', 'pub_time',
                 'price', 'station', 'rent']
    features.drop(columns=drop_cols, inplace=True)
    features = features.convert_dtypes()
    return features
    


df = bind_df()
stations_df = pd.read_csv(STATIONS_DF_PATH, index_col=0)

remove_special_chars()
remove_nonmoscow()
cols_from_parsed()

df = pd.merge(df, stations_df[['mcc', 'circle', 'center_distance']],
              how='left', left_on='matched_station', right_index=True)
features = form_features(df)
features.to_csv(FEATURES_SAVE_PATH)