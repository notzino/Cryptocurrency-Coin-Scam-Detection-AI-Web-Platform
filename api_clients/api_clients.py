import requests
import hashlib
import hmac
import time
import praw
from datetime import datetime, timedelta, timezone
from dateutil import parser
from config import (
    thenewsapi_token,
    blockchair_api_key,
    coinmarketcap_api_key,
    reddit_client_id,
    reddit_client_secret,
    reddit_user_agent,
    twitter_consumer_key,
    twitter_consumer_secret,
    twitter_access_token,
    twitter_access_token_secret,
)
import json
import yfinance as yf
from twarc import Twarc2, expansions
import os

class CryptoDataClient:
    @staticmethod
    def get_latest_data(symbol):
        try:
            yf_symbol = symbol if symbol.endswith('-USD') else symbol + "-USD"
            data = yf.download(tickers=yf_symbol, period='1d', interval='1m', timeout=10)

            if data.empty:
                print(f"No data found for symbol: {symbol}")
                return [], []

            latest = data.iloc[-1]
            current_time = latest.name.isoformat()
            prices = [[current_time, latest['Close']]]
            volumes = [[current_time, latest['Volume']]]
            return prices, volumes
        except Exception as e:
            print(f"Error in get_latest_data for {symbol}: {e}")
            return [], []

    @staticmethod
    def get_historical_data(symbol):
        periods = ['1y', 'max']
        for period in periods:
            try:
                yf_symbol = symbol if symbol.endswith('-USD') else symbol + "-USD"
                data = yf.download(tickers=yf_symbol, period=period, interval='1wk', timeout=10)
                if data.empty:
                    print(f"No data found for symbol: {symbol} for period: {period}")
                    continue

                historical_data = []
                for index, row in data.iterrows():
                    timestamp = index.strftime('%Y-%m-%d %H:%M:%S')
                    price = row['Close']
                    volume = row['Volume']
                    historical_data.append([timestamp, price, volume])
                if historical_data:
                    return historical_data
            except yf.shared.exceptions.YFInvalidPeriodError as e:
                print(f"Invalid period error for {symbol} for period {period}: {e}")
            except yf.shared.exceptions.YFPricesMissingError as e:
                print(f"Prices missing error for {symbol}: {e}")
            except Exception as e:
                print(f"Error in get_historical_data for {symbol} for period {period}: {e}")
        return []

class NewsAPIClient:
    @staticmethod
    def get_news(query):
        url = 'https://api.thenewsapi.com/v1/news/all'
        max_articles = 100
        articles = []
        unique_urls = set()
        current_date = datetime.now(timezone.utc)
        start_date = current_date - timedelta(days=365 * 5)

        params = {
            'api_token': thenewsapi_token,
            'search': query,
            'published_after': start_date.strftime('%Y-%m-%d'),
            'published_before': current_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sort': 'published_on',
            'limit': 50,
            'page': 1
        }

        try:
            while len(articles) < max_articles:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if 'data' in data:
                    for article in data['data']:
                        if article['url'] not in unique_urls:
                            unique_urls.add(article['url'])
                            published_at = parser.parse(article['published_at']).astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                            articles.append({
                                'title': article['title'],
                                'description': article['description'],
                                'url': article['url'],
                                'published_at': published_at
                            })
                            if len(articles) >= max_articles:
                                break
                    if len(data['data']) < params['limit']:
                        break
                    params['page'] += 1
                else:
                    break
        except requests.RequestException as e:
            print(f"Error fetching news for {query}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        print(f"Total articles fetched for {query}: {len(articles)}")
        return articles[:max_articles]

    @staticmethod
    def get_news_for_scam(coin_name):
        queries = [
            f"{coin_name} scam",
            f"{coin_name} fraud",
            f"{coin_name} Ponzi scheme",
            f"{coin_name} founders arrested",
            f"{coin_name} investors losses"
        ]
        all_articles = []
        for query in queries:
            articles = NewsAPIClient.get_news(query)
            all_articles.extend(articles)
        return all_articles

    @staticmethod
    def get_news_for_scam(coin_name):
        queries = [
            f"{coin_name} scam",
            f"{coin_name} fraud",
            f"{coin_name} Ponzi scheme",
            f"{coin_name} founders arrested",
            f"{coin_name} investors losses"
        ]
        all_articles = []
        for query in queries:
            articles = NewsAPIClient.get_news(query)
            all_articles.extend(articles)
        return all_articles

class BlockchairClient:
    BASE_URL = "https://api.blockchair.com/"
    MAX_REQUESTS_PER_MINUTE = 30
    REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE

    def __init__(self):
        self.last_request_time = 0
        self.request_count = 0

    def _rate_limit(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_request_time
        if self.request_count >= self.MAX_REQUESTS_PER_MINUTE:
            if elapsed_time < 60:
                time_to_wait = 60 - elapsed_time
                print(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds.")
                time.sleep(time_to_wait)
            self.request_count = 0
            self.last_request_time = time.time()
        else:
            if elapsed_time < self.REQUEST_INTERVAL:
                time.sleep(self.REQUEST_INTERVAL - elapsed_time)
            self.request_count += 1
            self.last_request_time = time.time()

    def get_transactions(self, blockchain, address):
        self._rate_limit()
        url = f"{self.BASE_URL}{blockchain}/dashboards/address/{address}?transaction_details=true"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and address in data['data']:
                    return data['data'][address]['transactions']
            print(f"Blockchair API request failed for transactions of address {address} with status code {response.status_code}")
        except requests.RequestException as e:
            print(f"Error fetching transactions for {address}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return []

    def validate_address(self, address):
        valid_prefixes = (
            '1', '3', 'bc1', '0x', 'addr1', 'GitYuc', 'DEgD', 'ADNb',
            '14KX', '15qr', '16ZL', '14gA', 'M8T1', 'GAHJ', 'GDQP',
            'cosmos', 'AEBR', 'AEUU', 'TQjq', 'TQUsh', 'f1a', 'f2a'
        )
        return any(address.startswith(prefix) for prefix in valid_prefixes)

class HistoricalBlockchainDataClient:
    @staticmethod
    def get_historical_transactions(symbol):
        return []

class TwitterClient:
    def __init__(self):
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if not self.bearer_token:
            raise ValueError("TWITTER_BEARER_TOKEN environment variable not set")

        self.twarc_client = Twarc2(bearer_token=self.bearer_token)

    def get_tweets(self, query, max_results=100):
        tweets = []
        try:
            start_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat("T") + "Z"
            search_results = self.twarc_client.search_recent(query=query, max_results=max_results, start_time=start_time)
            for page in search_results:
                result = expansions.flatten(page)
                for tweet in result:
                    tweets.append({
                        'created_at': tweet['created_at'],
                        'text': tweet['text'],
                        'url': f"https://twitter.com/user/status/{tweet['id']}"
                    })
        except Exception as e:
            print(f"Error fetching tweets with twarc: {e}")

        return tweets


class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )

    def get_reddit_posts(self, query, start_date, end_date, limit=1000):
        posts = []
        subreddit = self.reddit.subreddit('cryptocurrency')
        print(f"Searching for posts in subreddit 'cryptocurrency' with query '{query}' from {start_date} to {end_date}")

        try:
            for submission in subreddit.search(query, time_filter='all', limit=limit):
                created_at = datetime.fromtimestamp(submission.created_utc, timezone.utc)
                # print(f"Found post: {submission.title}, created at: {created_at}")
                if start_date.astimezone(timezone.utc) <= created_at <= end_date.astimezone(timezone.utc):
                    posts.append({
                        'title': submission.title,
                        'created_at': created_at,
                        'selftext': submission.selftext,
                        'num_comments': submission.num_comments,
                        'score': submission.score
                    })
        except Exception as e:
            print(f"Error fetching Reddit posts for {query}: {e}")

        print(f"Total posts found: {len(posts)}")
        return posts

