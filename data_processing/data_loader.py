import pandas as pd
from database.database import Database


def load_data_from_db():
    db = Database()
    coins_data = db.fetchall("SELECT id, coin_name, symbol, is_scam FROM cryptocurrencies")
    historical_data = db.fetchall("SELECT coin_id, timestamp, price, volume FROM historical_data")
    news_data = db.fetchall("SELECT coin_id, title, description, url, published_at FROM news_articles")
    social_media_data = db.fetchall("SELECT coin_id, platform, content, created_at, num_comments, score FROM social_media")
    wallet_data = db.fetchall("SELECT coin_id, address, transaction_data FROM wallet_data")
    transactions_data = db.fetchall("SELECT wallet_data_id, block_id, transaction_hash, time, balance_change, created_at, updated_at FROM wallet_transactions")
    db.close()

    coins_df = pd.DataFrame(coins_data, columns=['id', 'coin_name', 'symbol', 'is_scam']).drop_duplicates()
    historical_df = pd.DataFrame(historical_data, columns=['coin_id', 'timestamp', 'price', 'volume']).drop_duplicates()
    news_df = pd.DataFrame(news_data, columns=['coin_id', 'title', 'description', 'url', 'published_at']).drop_duplicates()
    social_media_df = pd.DataFrame(social_media_data, columns=['coin_id', 'platform', 'content', 'created_at', 'num_comments', 'score']).drop_duplicates()
    wallet_data_df = pd.DataFrame(wallet_data, columns=['coin_id', 'address', 'transaction_data']).drop_duplicates()
    transactions_df = pd.DataFrame(transactions_data, columns=['wallet_data_id', 'block_id', 'transaction_hash', 'time', 'balance_change', 'created_at', 'updated_at']).drop_duplicates()

    return coins_df, historical_df, news_df, social_media_df, wallet_data_df, transactions_df
