from datetime import datetime, timedelta
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers
from snscrape.base import ScraperException

import glob, os, re
import numpy as np
import pandas as pd
import time



def clean_tweets(df):

    drop_idxs = list()
    tweets = df['tweet'].tolist()
    clean_tweets = list()

    for index, tweet in enumerate(tweets):

        if ('Apple' not in tweet) and ('AAPL' not in tweet):
            drop_idxs.append(index)

        tweet_lines = tweet.split('\n')
        tweet = ' '.join([line for line in tweet_lines if not line.isnumeric()])
        clean_tweets.append(tweet)

    df['tweet'] = clean_tweets
    df.drop(drop_idxs, inplace=True)


def filter_english(df):

    drop_idxs = list()
    tweets = df['tweet'].tolist()

    for index, tweet in enumerate(tweets):

        try:

            lang = detect(tweet)

        except LangDetectException:

            drop_idxs.append(index)

        if lang != 'en':

            drop_idxs.append(index)

    df.drop(list(set(drop_idxs)), inplace=True)



# def scrape_tweets(ticker, comp, start_date, end_date):

#     search_string = f"${ticker} {comp}"

#     start_time = time.time()

#     data = scrape(words=[search_string], since=start_date, until=end_date,
#                   interval=1, headless=False, display_type="Top", 
#                   save_images=False, resume=False, proximity=False)

#     end_time = time.time()

#     print(f'Seconds required to scrape tweets from {start_date} to {end_date}: {end_time-start_time}')



def scrape_twitter_by_day(keyword, day):

    curr_dir = os.getcwd()

    ticker = keyword.split(' ')[0]
    ticker_dir = f'{curr_dir}/{ticker}_tweets/'

    if not os.path.exists(ticker_dir):
        os.makedirs(ticker_dir)

    file_path = f'{ticker_dir}/{ticker}_tweets_on_{day}.json'

    date = datetime(*[int(item) for item in day.split('-')])
    next_day = (date + timedelta(days=1)).strftime('%Y-%m-%d')

    snscrape_str = str()
    snscrape_str += f"snscrape --jsonl " 
    snscrape_str += f"twitter-search \'{keyword} since:{day} until:{next_day}\'"
    snscrape_str += f" > \'{file_path}\'"



    if not os.path.exists(file_path):


        print(f'Gathering tweets for {ticker} on {day}...')

        start_time = time.time()

        try:

            os.system(snscrape_str)

        except ScraperException:

            # time.sleep(0.1)
            # os.system(snscrape_str)
            print('ScraperException!')

        end_time = time.time()

        print(f'Seconds required to pull tweets for {day}: {end_time-start_time}\n')

    df = pd.read_json(file_path, lines=True)
    print(df.columns.tolist())


def scrape_twitter_between_dates(keyword, start_date, end_date):

    date_format = '%Y-%m-%d'
    curr_datetime = datetime.strptime(start_date, date_format)
    end_datetime = datetime.strptime(end_date, date_format)

    while curr_datetime < end_datetime:

        date = datetime.strftime(curr_datetime, date_format)
        scrape_twitter_by_day(keyword, date)
        curr_datetime += timedelta(days=1)







