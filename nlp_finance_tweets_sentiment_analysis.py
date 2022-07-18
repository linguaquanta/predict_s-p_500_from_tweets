from bs4 import BeautifulSoup as BS, SoupStrainer
from bs4.element import Comment

from datetime import datetime, timedelta
from utils import (search_twitter_between_dates,
                   filter_verified_tweets, filter_raw_tweets,
                   finbert_analyze, convert_dates_to_y2k_units)

from joblib import Parallel, delayed
from pathlib import Path
from pprint import pprint
from pytz import timezone
from scipy import signal
from sklearn.metrics import classification_report
from statsmodels.tsa.stattools import grangercausalitytests as gct
from textblob import TextBlob
from transformers import AutoModelForSequenceClassification

import emoji, glob, json, logging
import lxml, lxml.html, os, os.path, sys, time
import pickle, re, requests, scrapy, shutil
import torch, urllib.parse, urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import httplib2 as hlib, html5lib
import yfinance as yf


###################### finBERT ############################

from finBERT.finbert.finbert import tokenizer, model, 
import finBERT.finbert.utils as tools

###########################################################

# Eventually want to loop over a list of all tickers 
# for companies in the S&P 500 and construct a 
# market-cap-weighted sentiment analysis...

spx_tickers = pd.read_csv('SPX_tickers_by_market_cap.csv')
num_comps = 1

tickers = spx_tickers['Symbol'].tolist()[:num_comps]
weights = spx_tickers['Weight'].tolist()[:num_comps]

start_date = '2020-05-01'
end_date = '2020-05-10'
# end_date = '2021-12-29'

# for ticker in tickers:

ticker = 'AAPL'

curr_dir = os.getcwd()
ticker_dir = curr_dir + f'/{ticker}_tweets/'

if not os.path.exists(ticker_dir):
    os.makedirs(ticker_dir)

tweet_bank_csv = f'{ticker.lower()}_tweet_bank.csv'
stock_vals_csv = f'{ticker.lower()}_stock_vals.csv'


############### TOGGLES ##############

load_fin_data = 0

load_tweets = 0

check_missing = 0

load_dataframe = 0

run_finbert = 0

convert_dates_to_y2k = 0

######################################

if __name__ == '__main__':

    # Load financial data between specified dates...

    if bool(load_fin_data):

        fin_data = yf.download(tickers=ticker, start=start_date, 
                               end=end_date, interval='1h')

        fin_data.to_csv(stock_vals_csv)

        fin_data = pd.read_csv(stock_vals_csv, index_col=False)
        fin_cols = fin_data.columns.tolist()

        if 'Unnamed: 0' in fin_cols:
            if 'Date' in fin_cols:
                fin_data.drop(columns=['Unnamed: 0'], inplace=True)
            else:
                fin_data = fin_data.rename(columns={'Unnamed: 0': 'Date'})


        fin_dates = fin_data['Date'].tolist()


    # Use snscrape to pull all tweets between
    # the specified start and end dates...

    files = glob.glob(os.path.join(ticker_dir + '*.json'))
    tweet_dates = [file.split('_')[-1].split(' ')[0] for file in sorted(files)]

    if bool(load_tweets):

        search_twitter_between_dates(ticker, start_date, end_date)


    # if check_missing:

    #     missing_days = list(set([date.split(' ')[0] for date in fin_dates])-set(tweet_dates))

    #     if len(missing_days) > 0:

    #         print(f'{len(missing_days)} days with missing tweet data...')
    #         for day in missing_days:
    #             search_twitter_by_day(ticker, day)



    # Initialize and populate a dataframe with
    # content pulled from Twitter scrape above.

    ticker_df = pd.DataFrame(columns=['date', 'content'])


    if bool(load_dataframe):

        for file in sorted(files): 
            print(file)
            df = pd.read_json(file, lines=True)
            print(df)
            # # if'lang' not in df.columns.tolist():
            # #     print(df)
            # # df = filter_raw_tweets(df)
            # print(len(df))
            # ticker_df = ticker_df.append(df)


        ticker_df.to_csv(tweet_bank_csv)

    # Load dataframe from file above, filter for verified
    # users and convert emojis to text.

    ticker_df = pd.read_csv(tweet_bank_csv, index_col=False)
    ticker_df.drop(columns=['Unnamed: 0'], inplace=True)

    filter_verified_tweets(ticker_df)
    filter_raw_tweets(ticker_df)

    ticker_df.to_csv(tweet_bank_csv)

    ticker_df = pd.read_csv(tweet_bank_csv, index_col=False)
    ticker_df.drop(columns=['Unnamed: 0'], inplace=True)
    tweets = ticker_df['content'].tolist()
    num_tweets = len(tweets)

    # Create a new dataframe for storing only timestamps, tweets
    # and sentiment scores as calculated by finBERT (below).

    sent_df = pd.DataFrame(columns=["Date", "Tweet", "Sentiment"])
    sent_df_csv = f'{ticker.lower()}_tweets_{start_date}_to_{end_date}.csv'


    dates = ticker_df['date'].tolist()
    sent_df["Date"] = dates


    if bool(run_finbert):

        comp_scores = list()

        start_total_time = time.time()
        finbert_analyze(ticker, tweets, comp_scores, sent_df, sent_df_csv)
        end_total_time = time.time()

        total_time = end_total_time-start_total_time
        time_str = f'{total_time/60} minutes required to run finBERT on all {num_tweets} tweets...'
        print(time_str)        


        sent_df = pd.read_csv(sent_df_csv, index_col=False)
        if 'Unnamed: 0' in sent_df.columns.tolist():
            sent_df.drop(columns=['Unnamed: 0'], inplace=True)

    sent_dates = sent_df['Date'].tolist()
    fin_dates = fin_data['Date'].tolist()

    if bool(convert_dates_to_y2k):

        fin_data['Date'] = convert_dates_to_y2k_units(fin_dates)
        fin_data.to_csv(stock_vals_csv)
        sent_df['Date'] = convert_dates_to_y2k_units(sent_dates)
        sent_df.to_csv(sent_df_csv)







