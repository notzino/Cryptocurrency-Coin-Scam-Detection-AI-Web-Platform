import re
import logging
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from data_processing.data_loader import load_data_from_db

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
model = SentenceTransformer('all-MiniLM-L6-v2')


def preprocess_text(text):
    if pd.isnull(text):
        return ""
    try:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
        return ' '.join(words)
    except Exception as e:
        logging.error(f"Error preprocessing text: {text}. Error: {e}")
        return ""


def handle_missing_values(df):
    logging.info(f"Before handling missing values: {df.isnull().sum()}")
    df = df.ffill()
    df = df.bfill()
    logging.info(f"After handling missing values: {df.isnull().sum()}")
    return df


def extract_features(text):
    try:
        if not text.strip():
            return [0] * 384
        return model.encode(text).tolist()
    except Exception as e:
        logging.error(f"Error extracting features: {text}. Error: {e}")
        return [0] * 384


def calculate_volatility(prices):
    """Calculate the volatility of price changes."""
    try:
        prices = [float(p) for p in prices if isinstance(p, (int, float, str)) and str(p).replace('.', '', 1).isdigit()]
        if len(prices) == 0:
            return 0
        return np.std(prices)
    except Exception as e:
        logging.error(f"Error calculating volatility: {e}")
        return 0


def calculate_volume_changes(volumes):
    """Calculate the change in volume."""
    try:
        volumes = [float(v) for v in volumes if
                   isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit()]
        if len(volumes) == 0:
            return 0
        return np.max(volumes) - np.min(volumes)
    except Exception as e:
        logging.error(f"Error calculating volume changes: {e}")
        return 0


def extract_sentiment(text):
    """Extract sentiment polarity from text."""
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as e:
        logging.error(f"Error extracting sentiment: {text}. Error: {e}")
        return 0


def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stop words
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())  # Lemmatization
    return text
def clean_historical_data(historical_df):
    historical_df.dropna(inplace=True)
    historical_df['price'] = pd.to_numeric(historical_df['price'], errors='coerce')
    historical_df['volume'] = pd.to_numeric(historical_df['volume'], errors='coerce')
    historical_df.dropna(inplace=True)

    scaler = StandardScaler()
    historical_df[['price', 'volume']] = scaler.fit_transform(historical_df[['price', 'volume']].copy())

    return historical_df


def clean_news_data(news_df):
    news_df.dropna(inplace=True)

    if 'description' in news_df.columns:
        news_df['cleaned_content'] = news_df['description'].apply(clean_text)
    else:
        raise KeyError("Column 'description' does not exist in news_df")

    return news_df


def clean_social_media_data(social_media_df):
    if 'content' not in social_media_df.columns:
        social_media_df['content'] = social_media_df['text']

    social_media_df['cleaned_content'] = social_media_df['content'].apply(clean_text)
    return social_media_df


if __name__ == "__main__":
    coins_df, historical_df, news_df, social_media_df, wallet_data_df, transactions_df = load_data_from_db()

    sample_text = "This is a sample text with a URL: https://example.com and some special characters!!!"
    processed_text = preprocess_text(sample_text)
    print(f"Processed text: {processed_text}")

    sample_prices = ["10.5", "11", "NaN", "13", "bad_data", 14.5]
    volatility = calculate_volatility(sample_prices)
    print(f"Volatility: {volatility}")

    sample_volumes = ["1000", "1500", "NaN", "2000", "bad_data", 2500]
    volume_change = calculate_volume_changes(sample_volumes)
    print(f"Volume change: {volume_change}")

    sentiment = extract_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")

    sample_df = pd.DataFrame({
        'A': [1, 2, 'bad_data', 4],
        'B': [5, 'NaN', 7, 8],
        'C': [9, 10, 11, 'NaN']
    })
    clean_df = handle_missing_values(sample_df)
    print(f"Clean DataFrame:\n{clean_df}")

    features = extract_features(sample_text)
    print(f"Features: {features}")

    cleaned_historical_df = clean_historical_data(historical_df)
    cleaned_news_df = clean_news_data(news_df)
    cleaned_social_media_df = clean_social_media_data(social_media_df)
