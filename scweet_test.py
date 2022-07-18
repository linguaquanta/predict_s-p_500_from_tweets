from datetime import datetime, timedelta
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers

import glob, os, re
import numpy as np
import pandas as pd


############################# FUNCTIONS ########################

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


###############################################################

start_date = "2021-12-29"
end_date = "2021-12-30"

cashtag = '$AAPL'
comp_name = 'Apple'

search_string = f"{cashtag}%20AND%20{comp_name}"

scrape_data = 0

if bool(scrape_data):
	data = scrape(words=[search_string], since=start_date, until=end_date,
				  interval=1, headless=False, display_type="Top", 
				  save_images=False, resume=False, proximity=False)

curr_dir = os.getcwd()
out_dir = os.path.join(f'{curr_dir}/outputs/')
files = sorted(glob.glob(os.path.join(f'{out_dir}/*.csv')))
filename = files[0].split('/')[-1]

df = pd.read_csv(f'{out_dir}/{filename}')
df = df.rename(columns={'Timestamp': 'date', 'Embedded_text': 'tweet'})


filter_english(df)
clean_tweets(df)
df = df.reset_index()


date_format = '%Y-%m-%d %H:%M:%S'
dates = [datetime.strptime(' '.join(item.split('T'))[:-5], date_format) for item in df['date'].tolist()]
df['date'] = dates
df.sort_values(by='date', inplace=True)
print(df[['date', 'tweet']])





