# -*- coding: utf-8 -*-
from datetime import date, datetime
import pathlib
import pickle
import sys
import time

from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import requests
from scipy.stats import expon, norm



PROJECT_FOLDER_PATH = pathlib.Path().resolve().parents[1]
RAW_FLATS_FOLDER_PATH = PROJECT_FOLDER_PATH / 'data/raw'

BREAK_TIME = 600
TAGS = {'text_tags': {
                      'header': ('a', {'class': "snippet-link"}),
                      'address': ('span', {'class': "item-address__string"}),
                      'station': ('span', "item-address-georeferences-item__content"),
                      'distance': ('span', "item-address-georeferences-item__after"),
                      'commission': ('span', {'class': "snippet-price-commission"}),
                      'published': ('div', {'class': "snippet-date-info"})
                     },
        'attr_tags': {
                      'ref': ('a', {'class': "snippet-link"}, 'href'),
                      'price': ('meta', {'itemprop': 'price'}, 'content')
                     }
        }

QUERY = 'https://www.avito.ru/moskva/kvartiry/sdam/na_dlitelnyy_srok-ASgBAgICAkSSA8gQ8AeQUg?f=ASgBAQICAkSSA8gQ8AeQUgFAzAhkllmUWZJZjFmOWZBZ&p={}'



def count_query_pages(query: str) -> int:
    first_page = query.format('1')
    first_response = requests.get(first_page, 
                                  headers={'User-Agent': UserAgent().chrome})
    first_source = first_response.content
    first_soup = BeautifulSoup(first_source, "lxml")
    pagination_soup = first_soup.find_all('span', {'class': 'pagination-item-1WyVp'})
    # print(pagination_soup)
    num_pages_soup = pagination_soup[-2]
    num_pages = int(num_pages_soup.text)
    print('Number of pages: ', num_pages)
    return num_pages

def count_missing_params(flats: list) -> dict:
    missing_keys = list(TAGS['text_tags'].keys()) + list(TAGS['attr_tags'].keys())
    missing = dict.fromkeys(missing_keys, 0) 
    for flat in flats:
        for key in missing:
            if flat[key] is None:
                missing[key] += 1
    return(missing)

def save_pickle(flats: list, parsing_date: datetime.date) -> None:
    flats_file = 'flats_{}.pickle'.format(parsing_date)
    sys.setrecursionlimit(200000)
    with open(RAW_FLATS_FOLDER_PATH / flats_file, 'wb') as f:
        pickle.dump(flats, f)



num_pages = count_query_pages(QUERY)
num_pages = 1
time.sleep(expon.rvs(37, 9))

parsing_date = date.today()
query_soup = []
access_time = []
for page_iter in range(num_pages, 0, -1):
    page_link = QUERY.format(page_iter)
    page_response = requests.get(page_link, 
                                 headers={'User-Agent': UserAgent().chrome})
    page_source = page_response.content
    print('{}/{}'.format(num_pages-page_iter+1, num_pages), page_response, page_link)
    if (page_response.status_code != 200) or (page_source is None):
        print('Connection aborted. Pause for {} seconds'.format(BREAK_TIME))
        time.sleep(BREAK_TIME)
        page_link = QUERY.format(page_iter)
        page_response = requests.get(page_link, 
                                     headers={'User-Agent': UserAgent().chrome})
    access_time.append(datetime.now())
    page_source = page_response.content
    page_soup = BeautifulSoup(page_source, "lxml")
    page_descriptions_soup = page_soup.find_all('div', {'class': "description"})
    query_soup.append(page_descriptions_soup)
    time.sleep(expon.rvs(28, 9) + norm.rvs(5, 7))
    page_iter = page_iter + 1
print('len(access time): ', len(access_time))
print('len(query soup): ', len(query_soup))

flats = []
for page_iter, page in enumerate(query_soup):
    for description in page:
        flat_soup = {}
        for key in TAGS:
            for param in TAGS[key]:
                flat_soup[param] = description.find(TAGS[key][param][0], TAGS[key][param][1])
        flat_params = {}
        for param in TAGS['text_tags']:
            if flat_soup[param] is not None:
                flat_params[param] = flat_soup[param].text
            else:
                flat_params[param] = None
        for param in TAGS['attr_tags']:
            if flat_soup[param] is not None:
                flat_params[param] = flat_soup[param][TAGS['attr_tags'][param][2]]
            else:
                flat_params[param] = None
        flat_params['parsing_date'] = access_time[page_iter]
        flats.append(flat_params)

print('missing:', count_missing_params(flats))
print('len(flats): ', len(flats))
print('unique links', len(set(flat['ref'] for flat in flats)))
# save_pickle(flats, parsing_date)