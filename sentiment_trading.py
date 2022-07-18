from bs4 import BeautifulSoup as BS
from bs4 import SoupStrainer
from bs4.element import Comment

from datetime import datetime, timedelta
from joblib import Parallel, delayed
from pathlib import Path
from pprint import pprint
from pytz import timezone
from scipy import signal
from sklearn.metrics import classification_report
from statsmodels.tsa.stattools import grangercausalitytests as gct
from textblob import TextBlob
from transformers import AutoModelForSequenceClassification

import emoji
import glob
import html5lib
import httplib2 as hlib
import logging
import lxml, lxml.html
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pandas as pd
import pickle
import re
import requests
import scipy.stats as stats
import shutil
import statsmodels.api as sm
import sys
import time
import torch
import urllib.parse, urllib.request
import yfinance as yf


# to provide context for finBERT's performance in
# properly identifying the sentiment scores for tweets
# we can easily also evaluate sentiment using NLTK

###################### finBERT ############################

from finBERT.finbert.finbert import *
import finBERT.finbert.utils as tools

##################### NLTK VADER ##########################

from nltk.sentiment.vader import SentimentIntensityAnalyzer


########################### FUNCTIONS ######################################

def filter_rows_by_col_cond(df, filter_col):
    df.drop(df[df[filter_col]==0].index, inplace=True)

def emoji_to_text(x):
    return emoji.demojize(x)

def clean_emoji_text(tweet):

    for item in re.findall(":(.*?):", tweet):
        tweet = tweet.replace(f':{item}:', ' '.join(item.split('_')))

    tweet = tweet.replace(':', ' ').replace('_', ' ')
    tweet = tweet.replace('#', '').replace('$', '')
    tweet = tweet.replace('https', '').replace('\n', ' ')

    tweet = tweet.split(' ')
    for word in tweet:
        if "//t.co/" in word:
            tweet.remove(word)

    tweet = ' '.join(tweet).rstrip()

    return tweet

# test search method without --max-results option   
# and replace since & until with a single date

def search_twitter_by_day(keyword, day):

    print(f'Gathering tweets for {day}...')

    date = datetime(*[int(item) for item in day.split('-')])
    next_day = (date + timedelta(days=1)).strftime('%Y-%m-%d')

    curr_dir = os.getcwd()

    snscrape_str = str()
    snscrape_str += f"snscrape --jsonl " 
    snscrape_str += f"twitter-search \'{keyword} since:{day} until:{next_day}\'"
    snscrape_str += f" > \'{curr_dir}/SPX_tweets/{keyword[3:]}_tweets_on_{day}.json\'"

    start_time = time.time()
    os.system(snscrape_str)
    end_time = time.time()

    print(f'Seconds required to pull tweets for {day}: {end_time-start_time}\n')

def search_twitter_between_dates(keyword, start_date, end_date):

    date_format = '%Y-%m-%d'
    curr_datetime = datetime.strptime(start_date, date_format)
    end_datetime = datetime.strptime(end_date, date_format)

    while curr_datetime < end_datetime:

        date = datetime.strftime(curr_datetime, date_format)
        search_twitter_by_day(keyword, date)
        curr_datetime += timedelta(days=1)

def filter_raw_tweets(df):

    hashtags = df['hashtags'].tolist()
    languages = df['lang'].tolist()

    has_hashtags = list(np.where(np.array([len(item) if item else 0 for item in hashtags]) > 0, 1, 0))
    is_english = [int(item=='en') for item in languages]

    df['has_hashtags'] = has_hashtags
    df['is_english'] = is_english

    for filter_col in ['has_hashtags', 'is_english']:
        filter_rows_by_col_cond(df, filter_col)

    df['content'] = df['content'].apply(lambda x: emoji_to_text(x))
    df['content'] = df['content'].apply(lambda x: clean_emoji_text(x))
    df['content'] = df['content'].apply(lambda x: x.replace('amp;', ''))
    df['content'] = df['content'].apply(lambda x: x.replace('SPX', 'S&P 500'))

    # save only dates and tweets
    cols = df.columns.tolist()
    cols.remove('date')
    cols.remove('content')
    drop_cols = cols
    df = df.drop(columns=drop_cols)

    return df

def sent_analyze_and_append(tweet, index, num_tweets, comp_scores):

    print(f'Running finBERT on tweet #{index+1} out of {num_tweets}...')

    start_tweet = time.time()

    text = [tweet]
    inputs = tokenizer(text, padding = True, truncation = True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    positive = predictions[:, 0].tolist()
    negative = predictions[:, 1].tolist()
    # neutral = predictions[:, 2].tolist()

    table = {'Headline': text,
             'Positive': positive,
             'Negative': negative}
          
    df = pd.DataFrame(table, columns=["Headline", "Positive", "Negative"])

    comp_score = df['Positive'].values[0]-df['Negative'].values[0]
    print(f'comp score: {comp_score}')
    # comp_scores.append(comp_score)
    end_tweet = time.time()
    tweet_time = end_tweet-start_tweet
    print(f'{tweet_time} seconds required to run finBERT on tweet #{index+1}...\n')
    comp_scores.append(comp_score)

def clean_tweets_url(tweets):

    print(f'Cleaning tweets of Twitter URLs and printing to file...')

    cleaned_tweets = list()
    # drop_indices = list()
    # need to grab indices to drop 'nan' (type==float) tweets
    for index, tweet in enumerate(tweets):

        if not (type(tweet)==float):
            words = tweet.split(' ')

            for word in words:

                if '//t.co' in word:
                    words.remove(word)

            tweet = ' '.join(words).rstrip()
            # tweets[index] = tweet
            cleaned_tweets.append(tweet)
        else:
            # drop_indices.append(index)
            # drop row in bank_df by index
            bank_df.drop([index], inplace=True)

    bank_df['content'] = cleaned_tweets
    bank_df.to_csv(spx_tweet_bank_csv)

def convert_dates_to_y2k_units(dates):

    num_dates = list()

    for date in dates:
        tweet_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z')
        weeks_since_y2k = (tweet_time-y2k).total_seconds()/(3600)
        num_dates.append(weeks_since_y2k-first_date_weeks_since_y2k)

    return num_dates

def plot_df(df, x, y, title="", xlabel='Time', ylabel='Value', dpi=100):

    plt.figure(figsize=(16,5), dpi=dpi)
    plt.scatter(x, y, color='tab:red', s=3)
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

def plot_dfs(df1, df2, x, y1, y2, title="", xlabel='Time', ylabel='Value', dpi=100):

    plt.figure(figsize=(16,5), dpi=dpi)

    plt.scatter(x, y1, color='tab:red', s=3)
    plt.scatter(x, y2, color='tab:blue', s=3)

    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

def crosscorr(datax, datay, lag=0, wrap=False):

    """ Lag-N cross correlation. Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """

    if wrap:

        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)

    else:

        return datax.corr(datay.shift(lag))

######################################################################################

# Download all tweets containing $SPX from 2020-12-24 to 2021-12-24 by day...

start_date = '2021-01-01'
end_date = '2021-12-24'
keyword = "%24SPX"
curr_dir = os.getcwd()
spx_dir = curr_dir + '/SPX_tweets/'
spx_tweet_bank_csv = 'spx_bank.csv'

toggle_load_tweets = False
if toggle_load_tweets:

    search_twitter_between_dates(keyword, start_date, end_date)

bank_df = pd.DataFrame(columns=['date', 'content'])
files = glob.glob(os.path.join(spx_dir + '*.json'))

toggle_load_dataframe = False
if toggle_load_dataframe:

    for file in sorted(files): 
        
        date = file.split('_')[-1][:-5]
        df = pd.read_json(file, lines=True)
        df = filter_raw_tweets(df)
        # print(df)
        bank_df = bank_df.append(df)


    bank_df.to_csv(spx_tweet_bank_csv)

# index_lim_test = 10
bank_df = pd.read_csv(spx_tweet_bank_csv)
tweets = bank_df['content'].tolist() #[:index_lim_test]
toggle_clean_url_tweets = False

if toggle_clean_url_tweets:
    clean_tweets_url(tweets)


# # finBERT for sentiment analysis of all tweets
# # need to time process when running over all tweets


# sys.path.append('..')
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# finbert_git_rep = "ProsusAI/finbert"
# project_dir = Path.cwd().parent
# pd.set_option('max_colwidth', None)
# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S', level = logging.ERROR)

# sent_df_pickle = 'spx_previous_year_tweets_2020-12-24_to_2021-12-24.p'
sent_df_csv = 'spx_previous_year_tweets_2020-12-24_to_2021-12-24.csv'

# sent_df = pd.DataFrame(columns=["Date", "Tweet", "Sentiment"])


# dates = bank_df['date'].tolist() #[:index_lim_test]
# sent_df["Date"] = dates

toggle_run_finbert = False

# start_total_time = time.time()
# num_tweets = len(tweets)


# if not os.path.exists(sent_df_pickle) or toggle_run_finbert:

#     comp_scores = list()
#     tokenizer = AutoTokenizer.from_pretrained(finbert_git_rep)
#     model = AutoModelForSequenceClassification.from_pretrained(finbert_git_rep)

#     # use multiprocessing to speed up this loop...
#     for index, tweet in enumerate(tweets):
#         sent_analyze_and_append(tweet, index, num_tweets, comp_scores)
    
#     # for : #[:index_lim_test]
        

#     sent_df["Tweet"] = tweets # [:index_lim_test]
#     sent_df["Sentiment"] = comp_scores
#     sent_df.to_pickle(sent_df_pickle)
#     sent_df.to_csv(sent_df_csv)

# end_total_time = time.time()
# total_time = end_total_time-start_total_time
# print(f'{total_time/60} minutes required to run finBERT on all {num_tweets} tweets...')

sent_df = pd.read_csv(sent_df_csv)
sent_df.drop(columns=['Unnamed: 0'], inplace=True)

eastern = timezone('US/Eastern')
utc = timezone('UTC')
y2k = eastern.localize(datetime(2000,1,1,0,0))
num_date_xmas_eve_2020 = 7663.993217592592


dates = sent_df['Date'].tolist()
sents = sent_df['Sentiment'].tolist()

first_date = utc.localize(datetime.strptime(sorted(dates)[0], '%Y-%m-%d %H:%M:%S+00:00'))
first_date_weeks_since_y2k = (first_date-y2k).total_seconds()/(3600)

# Dates are listed in *reverse* hourly order within days that are listed properly
# in increasing order from Xmas Eve 2020 to Xmas Eve 2021

sent_df['Date'] = convert_dates_to_y2k_units(dates)
sent_df = sent_df.sort_values(by=['Date'], ascending=True)


dates = sent_df['Date'].tolist()
unique_dates = np.unique(dates)
tweet_sep = [unique_dates[i+1]-unique_dates[i] for i in range(len(unique_dates)-1)]


spx_vals_csv = 'spx_vals.csv'
spx_df = pd.read_csv(spx_vals_csv)
spx_cols = spx_df.columns.tolist()

if 'Unnamed: 0' in spx_cols:
    spx_df = spx_df.rename(columns={"Unnamed: 0": "Date"})

spx_opens = np.array(spx_df['Open'].tolist())
spx_closes = np.array(spx_df['Close'].tolist())
spx_returns = (spx_closes-spx_opens).tolist()
spx_dates = spx_df['Date'].tolist()

spx_df['Return'] = spx_returns
spx_df['Date'] = convert_dates_to_y2k_units(spx_dates)

spx_return = spx_df[['Date', 'Return']]
spx_sent = sent_df[['Date', 'Sentiment']]


first_date = float(spx_return.iloc[0]['Date'])
second_date = float(spx_return.iloc[1]['Date'])

# print(spx_sent.loc[ (spx_sent['Date'] >= first_date) & (spx_sent['Date'] <= second_date)])

# Create new df with averaged sentiment scores over hours to match up with return data
new_df = pd.DataFrame(columns=['Date', 'Avg Sent'])
return_dates = spx_return['Date'].tolist()
new_df['Date'] = return_dates


avg_sents = list()
temp_df = spx_sent.loc[(spx_sent['Date'] <= return_dates[0])]
avg_sents.append(np.mean(temp_df['Sentiment']))

for index in range(len(return_dates)-1):

    temp_df = spx_sent.loc[(spx_sent['Date'] >= return_dates[index]) & (spx_sent['Date'] <= return_dates[index+1])]
    avg_sent = np.mean(temp_df['Sentiment'])
    avg_sents.append(avg_sent)
    # print([avg_sent, return_dates[index]])

new_df['Avg Sent'] = avg_sents
drop_idxs = new_df[new_df['Avg Sent'].isnull()==True].index.tolist()
new_df = new_df.drop(drop_idxs)
spx_return = spx_return.drop(drop_idxs)

first_hour = spx_return['Date'].tolist()[0]

new_df['Date'] = new_df['Date'].apply(lambda x: x-first_hour)
spx_return['Date'] = spx_return['Date'].apply(lambda x: x-first_hour)


# plot_df(spx_return, x=spx_return['Date'].tolist(), y=spx_return['Return'].tolist(), 
#         title='Hourly returns for the S&P 500 since 2020-12-24' )

sents = new_df['Avg Sent'].tolist()
returns = spx_return['Return'].tolist()

max_sent = np.max(sents)
max_return = np.max(returns)

# new_df['Avg Sent'] = new_df['Avg Sent'].apply(lambda x: x/max_sent)
# spx_return['Return'] = spx_return['Return'].apply(lambda x: x/max_return)

# plot_df(new_df[:100], x=new_df['Date'].tolist()[:100], y=new_df['Avg Sent'].tolist()[:100], 
#         title='Hourly Twitter sentiment about the S&P 500 since 2020-12-24' )

# cutoff = 24
# plot_dfs(new_df.iloc[:cutoff], spx_return.iloc[:cutoff], x=spx_return['Date'].tolist()[:cutoff], y1=new_df['Avg Sent'].tolist()[:cutoff],
#          y2=spx_return['Return'].tolist()[:cutoff], title='Normalized hourly Twitter sentiments and S&P 500 returns' )

data = pd.DataFrame(columns=['Avg Sent', 'Return'])

# norm_sents = new_df['Avg Sent'].tolist()
# norm_returns = spx_return['Return'].tolist()

data['Avg Sent'] = sents
data['Return'] = returns


# d1 = data['Avg Sent']
# d2 = data['Return']

# max_lag = 240
# for i in range(max_lag):
#     print(d1.corr(d2.shift(i)))

# cross_corr = [crosscorr(d1, d2, lag) for lag in range(1,24)]
# offset = np.ceil(len(cross_corr)/2)-np.argmax(cross_corr)
# f,ax=plt.subplots(figsize=(14,3))
# ax.plot(cross_corr)
# ax.axvline(np.ceil(len(cross_corr)/2),color='k',linestyle='--',label='Center')
# ax.axvline(np.argmax(cross_corr),color='r',linestyle='--',label='Peak synchrony')
# ax.set(title=f'Offset = {offset} frames\nAvg Sent leads <> Return leads',ylim=[.1,.31],xlim=[0,24], xlabel='Offset',ylabel='Pearson r')
# # ax.set_xticklabels([int(item-150) for item in ax.get_xticks()])
# plt.legend()
# plt.show()

# print(d2)
# print(d2.shift(1))