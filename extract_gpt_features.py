import os
import pandas as pd
import numpy as np
import re
import logging
import openai
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from data_processing.data_loader import load_data_from_db
from data_processing.preprocessing import preprocess_text, handle_missing_values, calculate_volatility, calculate_volume_changes, extract_sentiment
from config import openai_api_key
import time
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info('Downloading punkt')
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logging.info('Downloading stopwords')
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logging.info('Downloading wordnet')
        nltk.download('wordnet', quiet=True)


ensure_nltk_data()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

client = openai.OpenAI(api_key=openai_api_key)

def preprocess_dfs(news_df, social_media_df, wallet_data_df, transactions_df):
    """Preprocess all text data in the dataframes."""
    news_df['description'] = news_df['description'].apply(preprocess_text)
    social_media_df['content'] = social_media_df['content'].apply(preprocess_text)
    if not wallet_data_df.empty:
        wallet_data_df['transaction_data'] = wallet_data_df['transaction_data'].apply(preprocess_text)
    if not transactions_df.empty:
        transactions_df['transaction_hash'] = transactions_df['transaction_hash'].apply(preprocess_text)
    return news_df, social_media_df, wallet_data_df, transactions_df

def load_and_preprocess_data():
    """Load and preprocess data."""
    coins_df, historical_df, news_df, social_media_df, wallet_data_df, transactions_df = load_data_from_db()

    news_df = handle_missing_values(news_df)
    social_media_df = handle_missing_values(social_media_df)
    if not wallet_data_df.empty:
        wallet_data_df = handle_missing_values(wallet_data_df)
    if not transactions_df.empty:
        transactions_df = handle_missing_values(transactions_df)

    news_df, social_media_df, wallet_data_df, transactions_df = preprocess_dfs(news_df, social_media_df, wallet_data_df, transactions_df)

    return coins_df, historical_df, news_df, social_media_df, wallet_data_df, transactions_df

def extract_features_for_model(coins_df, historical_df, news_df, social_media_df, wallet_data_df=None, transactions_df=None):
    if wallet_data_df is None:
        wallet_data_df = pd.DataFrame()
    if transactions_df is None:
        transactions_df = pd.DataFrame()

    features = []

    for _, coin in coins_df.iterrows():
        coin_id = coin['id']

        try:
            coin_historical_data = historical_df[historical_df['coin_id'] == coin_id]
            prices = coin_historical_data['price'].values
            volumes = coin_historical_data['volume'].values
            volatility = calculate_volatility(prices)
            volume_changes = calculate_volume_changes(volumes)

            coin_news_data = news_df[news_df['coin_id'] == coin_id]
            news_sentiments = coin_news_data['description'].apply(extract_sentiment).values
            avg_news_sentiment = np.mean(news_sentiments) if len(news_sentiments) > 0 else 0

            coin_social_media_data = social_media_df[social_media_df['coin_id'] == coin_id]
            social_media_sentiments = coin_social_media_data['content'].apply(extract_sentiment).values
            avg_social_media_sentiment = np.mean(social_media_sentiments) if len(social_media_sentiments) > 0 else 0
            avg_num_comments = coin_social_media_data['num_comments'].mean() if 'num_comments' in coin_social_media_data.columns else 0
            avg_score = coin_social_media_data['score'].mean() if 'score' in coin_social_media_data.columns else 0
            avg_retweet_count = coin_social_media_data['retweet_count'].mean() if 'retweet_count' in coin_social_media_data.columns else 0
            avg_like_count = coin_social_media_data['like_count'].mean() if 'like_count' in coin_social_media_data.columns else 0

            avg_wallet_sentiment = 0
            if 'coin_id' in wallet_data_df.columns:
                coin_wallet_data = wallet_data_df[wallet_data_df['coin_id'] == coin_id]
                wallet_sentiments = coin_wallet_data['transaction_data'].apply(extract_sentiment).values
                avg_wallet_sentiment = np.mean(wallet_sentiments) if len(wallet_sentiments) > 0 else 0

            avg_transaction_sentiment = 0
            if 'wallet_data_id' in transactions_df.columns and not wallet_data_df.empty:
                coin_wallet_data = wallet_data_df[wallet_data_df['coin_id'] == coin_id]
                coin_transactions_data = transactions_df[transactions_df['wallet_data_id'].isin(coin_wallet_data.index)]
                transaction_sentiments = coin_transactions_data['transaction_hash'].apply(extract_sentiment).values
                avg_transaction_sentiment = np.mean(transaction_sentiments) if len(transaction_sentiments) > 0 else 0

            features.append({
                'coin_id': coin_id,
                'volatility': volatility,
                'volume_changes': volume_changes,
                'avg_news_sentiment': avg_news_sentiment,
                'avg_social_media_sentiment': avg_social_media_sentiment,
                'avg_num_comments': avg_num_comments,
                'avg_score': avg_score,
                'avg_retweet_count': avg_retweet_count,
                'avg_like_count': avg_like_count,
                'avg_wallet_sentiment': avg_wallet_sentiment,
                'avg_transaction_sentiment': avg_transaction_sentiment
            })
        except Exception as e:
            logging.error(f"Error processing coin_id {coin_id}: {e}")

    return pd.DataFrame(features)

def analyze_sentiments(news_df, social_media_df, wallet_data_df, transactions_df):
    """Analyze sentiments of news, social media, wallet data, and transactions."""
    news_df['sentiment'] = news_df['description'].apply(extract_sentiment)
    social_media_df['sentiment'] = social_media_df['content'].apply(extract_sentiment)
    if not wallet_data_df.empty:
        wallet_data_df['sentiment'] = wallet_data_df['transaction_data'].apply(extract_sentiment)
    if not transactions_df.empty:
        transactions_df['sentiment'] = transactions_df['transaction_hash'].apply(extract_sentiment)

    logging.info(f"News Sentiments: {news_df['sentiment'].describe()}")
    logging.info(f"Social Media Sentiments: {social_media_df['sentiment'].describe()}")
    if not wallet_data_df.empty:
        logging.info(f"Wallet Data Sentiments: {wallet_data_df['sentiment'].describe()}")
    if not transactions_df.empty:
        logging.info(f"Transaction Sentiments: {transactions_df['sentiment'].describe()}")

    return news_df, social_media_df, wallet_data_df, transactions_df

def summarize_data(df, text_column, max_entries=1):
    """Summarize data to reduce token size."""
    summary_list = df[text_column].sample(min(len(df), max_entries)).tolist()
    summary = ' '.join(summary_list)
    return ' '.join(summary.split()[:100])

def create_prompt(features_df_chunk, historical_df_chunk, summarized_news, summarized_social_media, summarized_wallet, summarized_transactions):
    """Create the prompt for the GPT-3.5 model."""
    return f"""
    Analyze the following data to identify key insights and correlations in detecting cryptocurrency scams. Look for significant words, time patterns, and correlations between historical data, news, social media, and wallet transactions.

    Features:
    {features_df_chunk.to_dict()}

    Historical Data:
    {historical_df_chunk.to_dict()}

    Summarized News Data:
    {summarized_news}

    Summarized Social Media Data:
    {summarized_social_media}

    Summarized Wallet Data:
    {summarized_wallet}

    Summarized Transaction Data:
    {summarized_transactions}

    Provide detailed insights.
    """

def count_tokens(text):
    """Count the number of tokens in a text."""
    return len(text.split())

def analyze_and_classify_with_gpt35(features_df, historical_df, news_df, social_media_df, wallet_data_df, transactions_df):
    responses = []
    chunk_size = 1
    max_requests_per_minute = 10
    delay_between_requests = 60 / max_requests_per_minute
    max_tokens_per_request = 3000

    for i in range(0, len(features_df), chunk_size):
        features_df_chunk = features_df.iloc[i:i + chunk_size]
        historical_df_chunk = historical_df.sample(n=chunk_size)

        summarized_news = summarize_data(news_df, 'description', max_entries=1)
        summarized_social_media = summarize_data(social_media_df, 'content', max_entries=1)
        summarized_wallet = summarize_data(wallet_data_df, 'transaction_data', max_entries=1)
        summarized_transactions = summarize_data(transactions_df, 'transaction_hash', max_entries=1)

        prompt = create_prompt(features_df_chunk, historical_df_chunk, summarized_news, summarized_social_media,
                               summarized_wallet, summarized_transactions)

        if count_tokens(prompt) > max_tokens_per_request:
            logging.error("Prompt too long, skipping this chunk.")
            continue

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            responses.append(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            responses.append("Error occurred while processing this chunk.")

        time.sleep(delay_between_requests)

    return "\n".join(responses)

def plot_sentiment_distribution(news_df, social_media_df, wallet_data_df, transactions_df, labels):
    """Plot sentiment distribution."""
    news_df['sentiment'] = news_df['sentiment'].fillna(0)
    social_media_df['sentiment'] = social_media_df['sentiment'].fillna(0)
    if not wallet_data_df.empty:
        wallet_data_df['sentiment'] = wallet_data_df['sentiment'].fillna(0)
    if not transactions_df.empty:
        transactions_df['sentiment'] = transactions_df['sentiment'].fillna(0)

    plt.figure(figsize=(12, 6))
    sns.histplot(data=news_df, x='sentiment', hue=labels, multiple="stack", bins=30)
    plt.title('Sentiment Distribution of News')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.histplot(data=social_media_df, x='sentiment', hue=labels, multiple="stack", bins=30)
    plt.title('Sentiment Distribution of Social Media')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    if not wallet_data_df.empty:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=wallet_data_df, x='sentiment', hue=labels, multiple="stack", bins=30)
        plt.title('Sentiment Distribution of Wallet Transactions')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

    if not transactions_df.empty:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=transactions_df, x='sentiment', hue=labels, multiple="stack", bins=30)
        plt.title('Sentiment Distribution of Transactions')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

def main():
    coins_df, historical_df, news_df, social_media_df, wallet_data_df, transactions_df = load_and_preprocess_data()
    features_df = extract_features_for_model(coins_df, historical_df, news_df, social_media_df, wallet_data_df, transactions_df)
    news_df, social_media_df, wallet_data_df, transactions_df = analyze_sentiments(news_df, social_media_df, wallet_data_df, transactions_df)
    labels = coins_df['is_scam']

    plot_sentiment_distribution(news_df, social_media_df, wallet_data_df, transactions_df, labels)

    gpt35_analysis = analyze_and_classify_with_gpt35(features_df, historical_df, news_df, social_media_df,
                                                     wallet_data_df, transactions_df)
    print(gpt35_analysis)

if __name__ == '__main__':
    main()
