from gevent import monkey
monkey.patch_all()

import joblib
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging

from flask_wtf.csrf import generate_csrf

from api_clients.api_clients import CryptoDataClient, NewsAPIClient, RedditClient
from data_processing.data_loader import load_data_from_db
from extract_gpt_features import extract_features_for_model
from insert_database_api import insert_historical_data, insert_news_data, insert_social_media_data, insert_coin_data
from database.database import Database
from data_processing.preprocessing import clean_historical_data, clean_news_data, clean_social_media_data
from xgboost import DMatrix
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import AgeLimit, TimeLimit, Retries

warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIMITED_DATA_THRESHOLD = 3

CRYPTO_KEYWORDS = ["cryptocurrency", "crypto", "blockchain", "coin", "token", "ICO", "DeFi", "NFT", "mining", "wallet"]

redis_broker = RedisBroker(
    url="redis://localhost:6379/0",
    middleware=[
        AgeLimit(max_age=3600000),  # 1 hour
        TimeLimit(time_limit=3600000),  # 1 hour
        Retries(max_retries=3)
    ]
)
dramatiq.set_broker(redis_broker)

def validate_coin(symbol):
    yf_symbol = symbol if symbol.endswith('-USD') else symbol + "-USD"
    coin = yf.Ticker(yf_symbol)
    try:
        info = coin.info
        if 'symbol' in info and info['symbol'] is not None:
            coin_name = info.get('shortName', symbol)
            logger.info(f"Validation successful for symbol: {symbol} with name: {coin_name}")
            return coin_name
        else:
            logger.error(f"No valid symbol found for symbol: {symbol}")
            return None
    except Exception as e:
        logger.error(f"Error: {e} for symbol: {symbol}, possibly invalid symbol")
        return None

def filter_crypto_related_content(content_list):
    filtered_content = []
    for content in content_list:
        if any(keyword in content['title'].lower() for keyword in CRYPTO_KEYWORDS) or \
                any(keyword in content.get('description', '').lower() for keyword in CRYPTO_KEYWORDS) or \
                any(keyword in content.get('content', '').lower() for keyword in CRYPTO_KEYWORDS):
            filtered_content.append(content)
    return filtered_content

def get_data_for_new_coin(symbol, progress_callback=None):
    logger.info(f"Validating coin: {symbol}")
    if progress_callback:
        progress_callback(10)

    coin_name = validate_coin(symbol)
    if not coin_name:
        logger.error(f"Coin {symbol} validation failed or it does not exist.")
        return None, None

    if progress_callback:
        progress_callback(20)

    try:
        logger.info(f"Fetching latest data for coin: {symbol}")
        latest_prices, latest_volumes = CryptoDataClient.get_latest_data(symbol)
        historical_data = CryptoDataClient.get_historical_data(symbol)

        if not historical_data:
            logger.error(f"No historical data found for symbol: {symbol}. The coin might be delisted.")
            return coin_name, {
                "latest_prices": latest_prices,
                "latest_volumes": latest_volumes,
                "historical_data": [],
                "news_data": [],
                "reddit_data": []
            }

        if progress_callback:
            progress_callback(30)

        coin_name_for_search = coin_name.replace("USD", "").strip()
        symbol_for_search = f"${symbol}"

        news_data_name = NewsAPIClient.get_news(coin_name_for_search)
        news_data_symbol = NewsAPIClient.get_news(symbol_for_search)
        news_data = filter_crypto_related_content(news_data_name + news_data_symbol)

        if progress_callback:
            progress_callback(40)

        reddit_data_name = RedditClient().get_reddit_posts(coin_name_for_search,
                                                           datetime.now(timezone.utc) - timedelta(days=365),
                                                           datetime.now(timezone.utc))
        reddit_data_symbol = RedditClient().get_reddit_posts(symbol_for_search,
                                                             datetime.now(timezone.utc) - timedelta(days=365),
                                                             datetime.now(timezone.utc))
        reddit_data = filter_crypto_related_content(reddit_data_name + reddit_data_symbol)

        if not news_data:
            logger.warning(f"No news articles found for coin: {coin_name}")
        if not reddit_data:
            logger.warning(f"No Reddit posts found for coin: {coin_name}")
    except Exception as e:
        logger.error(f"Error fetching data for symbol {symbol}: {e}")
        return None, None

    if progress_callback:
        progress_callback(50)

    return coin_name, {
        "latest_prices": latest_prices,
        "latest_volumes": latest_volumes,
        "historical_data": historical_data,
        "news_data": news_data,
        "reddit_data": reddit_data
    }

def prepare_features(coin_data, coins_df, historical_df, coin_name, symbol, progress_callback=None):
    if progress_callback:
        progress_callback(60)

    coin_id = coins_df['id'].max() + 1
    new_coin_df = pd.DataFrame([{
        "id": coin_id,
        "coin_name": coin_name,
        "symbol": symbol,
        "is_scam": False
    }]).drop_duplicates()

    coins_df = pd.concat([coins_df, new_coin_df], ignore_index=True).drop_duplicates()

    if coin_data['historical_data']:
        new_historical_df = pd.DataFrame(coin_data['historical_data'],
                                         columns=['Date', 'Close', 'Volume']).drop_duplicates()
    else:
        new_historical_df = pd.DataFrame(columns=['Date', 'Close', 'Volume']).drop_duplicates()
    new_historical_df['coin_id'] = coin_id

    historical_df = pd.concat([historical_df, new_historical_df], ignore_index=True).drop_duplicates()
    historical_df = clean_historical_data(historical_df)

    if progress_callback:
        progress_callback(70)

    news_data = coin_data['news_data']
    news_df = pd.DataFrame(news_data).drop_duplicates()
    news_df['coin_id'] = coin_id

    if 'description' not in news_df.columns:
        news_df['description'] = ''

    news_df = clean_news_data(news_df)

    reddit_data = coin_data['reddit_data']
    reddit_df = pd.DataFrame(reddit_data).drop_duplicates()
    reddit_df['coin_id'] = coin_id

    if 'content' not in reddit_df.columns:
        if 'text' in reddit_df.columns:
            reddit_df['content'] = reddit_df['text']
        elif 'body' in reddit_df.columns:
            reddit_df['content'] = reddit_df['body']
        else:
            reddit_df['content'] = ''

    reddit_df = clean_social_media_data(reddit_df)

    if progress_callback:
        progress_callback(80)

    features_df = extract_features_for_model(coins_df, historical_df, news_df, reddit_df)

    if progress_callback:
        progress_callback(90)

    return features_df[features_df['coin_id'] == coin_id]

def clean_historical_data(historical_df):
    if historical_df.empty:
        return historical_df

    if 'Close' in historical_df.columns and 'Volume' in historical_df.columns:
        scaler = StandardScaler()
        historical_df[['Close', 'Volume']] = scaler.fit_transform(historical_df[['Close', 'Volume']].copy())
    return historical_df

def insert_data_into_db(coin_id, coin_data):
    db = Database()
    try:
        insert_historical_data(db, coin_id, coin_data['historical_data'])
        insert_news_data(db, coin_id, coin_data['news_data'])
        insert_social_media_data(db, coin_id, 'Reddit', coin_data['reddit_data'])
    except Exception as e:
        logger.error(f"Error inserting data for coin ID {coin_id}: {e}")
    finally:
        db.close()

def coin_exists_in_db(symbol):
    db = Database()
    try:
        result = db.fetchone("SELECT id FROM cryptocurrencies WHERE symbol = %s", (symbol,))
        return result if result else None
    except Exception as e:
        logger.error(f"Error checking if coin exists in database: {e}")
        return None
    finally:
        db.close()

def predict_coin(symbol, coins_df, historical_df, news_df, social_media_df, wallet_data_df, transactions_df,
                 progress_callback=None):
    logger.info(f"Predicting coin for symbol: {symbol}")

    coin_name, coin_data = get_data_for_new_coin(symbol, progress_callback)
    if not coin_data:
        logger.error(f"Failed to get data for coin: {symbol}")
        return None, None, None, None

    logger.info(f"Fetched data for coin: {symbol}, name: {coin_name}")

    coins_df = coins_df.drop_duplicates()
    historical_df = historical_df.drop_duplicates()
    news_df = news_df.drop_duplicates()
    social_media_df = social_media_df.drop_duplicates()
    wallet_data_df = wallet_data_df.drop_duplicates()
    transactions_df = transactions_df.drop_duplicates()

    coin_id_result = coin_exists_in_db(symbol)
    logger.info(f"Checked if coin exists in database: {symbol}")

    if coin_id_result is None:
        logger.info(f"Coin {symbol} not found in database, inserting new data")
        db = Database()
        coin_id = insert_coin_data(db, coin_name, symbol, False)
        db.close()
        insert_data_into_db(coin_id, coin_data)
        features_df = prepare_features(coin_data, coins_df, historical_df, coin_name, symbol, progress_callback)
    else:
        coin_id = coin_id_result['id']
        insert_data_into_db(coin_id, coin_data)
        features_df = prepare_features(coin_data, coins_df, historical_df, coin_name, symbol, progress_callback)

    if features_df.empty or len(coin_data['news_data']) < LIMITED_DATA_THRESHOLD or len(
            coin_data['reddit_data']) < LIMITED_DATA_THRESHOLD or not coin_data['historical_data']:
        logger.warning(
            f"No features or limited data extracted for {coin_name} ({symbol}). Insufficient data, marking as uncertain.")
        return 0.5, coin_id, coin_name, 0

    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
    logger.info(f"Loading model from {model_path}")
    model_data = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    model = model_data['model']
    scaler = model_data['scaler']
    constant_filter = model_data['constant_filter']
    select_k_best = model_data['select_k_best']
    to_drop = model_data['to_drop']
    poly = model_data['poly']
    pca = model_data['pca']

    features_df_transformed = constant_filter.transform(scaler.transform(features_df.drop(columns=['coin_id'])))
    features_df_poly = poly.transform(select_k_best.transform(features_df_transformed))
    features_df_pca = pca.transform(features_df_poly)

    dmatrix = DMatrix(features_df_pca)
    prediction = model.predict(dmatrix)[0]

    # Confidence calculation
    confidence = (1 - abs(prediction - 0.5) * 2) * 100
    logger.info(f"Prediction: {prediction}, Confidence: {confidence}")

    return prediction, coin_id, coin_name, confidence

def save_prediction_to_db(symbol, coin_name, is_scam, coin_id):
    db = Database()
    try:
        logger.info(f"Saving prediction to database: coin_id={coin_id}, symbol={symbol}, coin_name={coin_name}, prediction={is_scam}")

        logger.info(f"prediction value type: {type(is_scam)}, prediction value: {is_scam}")

        db.execute("INSERT INTO predictions (coin_id, symbol, coin_name, prediction) VALUES (%s, %s, %s, %s)",
                   (coin_id, symbol, coin_name, is_scam))

        result = db.fetchone("SELECT prediction FROM predictions WHERE coin_id = %s AND symbol = %s", (coin_id, symbol))
        logger.info(f"Inserted prediction value from database: {result['prediction']}")

        logger.info(f"Prediction saved to database for {coin_name} ({symbol})")
    except Exception as e:
        logger.error(f"Error saving prediction to database: {e}")
    finally:
        db.close()

def update_scam_status(coin_id, is_scam):
    db = Database()
    try:
        is_scam_value = 1 if is_scam else 0
        logger.info(f"Updating scam status for coin ID {coin_id} to {is_scam_value}")

        db.execute("UPDATE cryptocurrencies SET is_scam = %s WHERE id = %s", (is_scam_value, coin_id))

        result = db.fetchone("SELECT is_scam FROM cryptocurrencies WHERE id = %s", (coin_id,))
        logger.info(f"Updated is_scam value from database: {result['is_scam']}")

        logger.info(f"Scam status updated for coin ID {coin_id}")
    except Exception as e:
        logger.error(f"Error updating scam status in database: {e}")
    finally:
        db.close()

def create_response(prediction, coin_id, coin_name, confidence):
    if prediction is None:
        return {'status': 'error', 'message': 'Invalid symbol or data retrieval issue.', 'coin_name': coin_name}
    if confidence == 0:
        return {'status': 'limited_data',
                'message': 'Insufficient data to make a reliable prediction. This coin is marked as a scam.',
                'coin_name': coin_name, 'confidence': f"{confidence:.0f}"}
    if prediction >= 0.5:
        return {'status': 'scam', 'message': 'This coin is predicted to be a scam.', 'coin_name': coin_name,
                'confidence': f"{confidence:.0f}"}
    return {'status': 'legit', 'message': 'This coin is not predicted to be a scam.', 'coin_name': coin_name,
            'confidence': f"{confidence:.0f}"}

def update_task_status(session_id, status):
    db = Database()
    try:
        logger.info(f"Updating task status for session_id {session_id} to {status}")
        db.execute("REPLACE INTO task_status (session_id, status) VALUES (%s, %s)", (session_id, status))
        logger.info(f"Task status updated successfully for session_id {session_id} to {status}")
    except Exception as e:
        logger.error(f"Error updating task status for session_id {session_id}: {e}")
    finally:
        db.close()

@dramatiq.actor
def predict_and_save_async(symbol, sid, csrf_token):
    from app import socketio
    try:
        logger.info(f"Starting predict_and_save for symbol: {symbol} with session_id: {sid}")
        update_task_status(sid, 'in_progress')

        def progress_callback(step):
            logger.info(f"Progress: {step}% for symbol: {symbol}")
            socketio.emit('progress', {'progress': step}, room=sid)
            logger.info(f"Emitting progress: {step}% to room {sid}")

        coins_df, historical_df, news_df, social_media_df, wallet_data_df, transactions_df = load_data_from_db()
        prediction, coin_id, coin_name, confidence = predict_coin(symbol, coins_df, historical_df, news_df, social_media_df, wallet_data_df, transactions_df, progress_callback)

        if prediction is None:
            logger.error(f"The coin ({symbol}) is not valid or data could not be retrieved.")
            response = {'status': 'error', 'message': f"The coin ({symbol}) is not valid or data could not be retrieved."}
            socketio.emit('prediction_result', response, room=sid)
            logger.info(f"Emitting prediction_result error to room {sid}")
            update_task_status(sid, 'failed')
            return

        is_scam = 1 if prediction >= 0.5 else 0
        save_prediction_to_db(symbol, coin_name, is_scam, coin_id)
        update_scam_status(coin_id, is_scam)

        response = create_response(prediction, coin_id, coin_name, confidence)
        logger.info(f"Prediction result prepared: {response}")
        socketio.emit('prediction_result', response, room=sid)
        logger.info(f"Emitting prediction_result success to room {sid}")

        update_task_status(sid, 'done')
    except Exception as e:
        logger.error(f"Error in predict_and_save_async: {str(e)}", exc_info=True)
        socketio.emit('prediction_result', {'status': 'error', 'message': str(e)}, room=sid)
        logger.info(f"Emitting prediction_result error to room {sid}")
        update_task_status(sid, 'failed')

if __name__ == '__main__':
    symbol = input("Enter the coin symbol: ").strip().upper()
    if symbol.endswith('-USD'):
        symbol = symbol[:-4]
    predict_and_save_async.send(symbol, 'dummy_sid', 'dummy_csrf_token')
