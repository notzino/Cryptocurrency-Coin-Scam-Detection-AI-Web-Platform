import os
import time
import json
from datetime import datetime, timedelta, timezone
from database.database import Database
from api_clients.api_clients import (
    CryptoDataClient,
    NewsAPIClient,
    BlockchairClient,
    RedditClient,
)
from coins import coins
import csv
import io

def insert_coin_data(db, coin_name, symbol, is_scam):
    try:
        result = db.fetchone("SELECT id FROM cryptocurrencies WHERE coin_name = %s AND symbol = %s", (coin_name, symbol))
        if result:
            return result['id']
        db.execute("""
            INSERT INTO cryptocurrencies (coin_name, symbol, is_scam)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE is_scam = VALUES(is_scam)
        """, (coin_name, symbol, is_scam))
        result = db.fetchone("SELECT LAST_INSERT_ID() AS id")
        if result and 'id' in result:
            return result['id']
        else:
            print(f"Failed to insert or retrieve coin ID for {coin_name} ({symbol})")
            raise Exception(f"Failed to insert or retrieve coin ID for {coin_name} ({symbol})")
    except Exception as e:
        print(f"Error inserting coin data for {coin_name}: {e}")
        raise

def insert_historical_data(db, coin_id, data, data_type='price'):
    for entry in data:
        try:
            if len(entry) == 2:
                timestamp, value = entry
                column = 'price' if data_type == 'price' else 'volume'
                db.execute(f"""
                    INSERT INTO historical_data (coin_id, timestamp, {column})
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE {column} = VALUES({column})
                """, (coin_id, timestamp, value))
            elif len(entry) == 3:
                timestamp, price, volume = entry
                db.execute(f"""
                    INSERT INTO historical_data (coin_id, timestamp, price, volume)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE price = VALUES(price), volume = VALUES(volume)
                """, (coin_id, timestamp, price, volume))
        except Exception as e:
            print(f"Error inserting historical data for coin ID {coin_id}: {e}")
            raise

def insert_news_data(db, coin_id, news_articles):
    for article in news_articles:
        try:
            db.execute("""
                INSERT INTO news_articles (coin_id, title, description, url, published_at)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE description = VALUES(description), url = VALUES(url), published_at = VALUES(published_at)
            """, (coin_id, article['title'], article['description'], article['url'], article['published_at']))
        except Exception as e:
            print(f"Error inserting news article for coin ID {coin_id}: {e}")
            raise

def insert_wallet_data(db, coin_id, address, transactions):
    try:
        db.execute("""
            INSERT INTO wallet_data (coin_id, address, transaction_data, created_at, updated_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE transaction_data = VALUES(transaction_data), updated_at = CURRENT_TIMESTAMP
        """, (coin_id, address, json.dumps(transactions)))
        wallet_data_id = db.fetchone("SELECT id FROM wallet_data WHERE coin_id = %s AND address = %s", (coin_id, address))['id']
        for transaction in transactions:
            db.execute("""
                INSERT INTO wallet_transactions (wallet_data_id, block_id, transaction_hash, time, balance_change)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE balance_change = VALUES(balance_change)
            """, (wallet_data_id, transaction['block_id'], transaction['hash'], transaction['time'], transaction['balance_change']))
    except Exception as e:
        print(f"Error inserting wallet data for coin ID {coin_id}, address {address}: {e}")
        raise

def insert_social_media_data(db, coin_id, platform, content):
    if not content:
        print(f"No content to insert for {platform}")
        return

    for post in content:
        try:
            created_at = post.get('created_at')
            selftext = post.get('selftext', '')
            num_comments = post.get('num_comments', 0)
            score = post.get('score', 0)

            if not created_at or not selftext:
                print(f"Skipping post due to missing 'created_at' or 'selftext': {post}")
                continue

            if isinstance(created_at, datetime):
                created_at = created_at.strftime('%Y-%m-%d %H:%M:%S')

            if not created_at:
                print(f"Skipping post with missing 'created_at': {post}")
                continue

            db.execute("""
                INSERT INTO social_media (coin_id, platform, content, created_at, num_comments, score)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE content = VALUES(content), created_at = VALUES(created_at), num_comments = VALUES(num_comments), score = VALUES(score)
            """, (coin_id, platform, selftext, created_at, num_comments, score))
        except Exception as e:
            print(f"Error inserting social media data for coin ID {coin_id}: {e}")
            raise

def process_wallet_transactions(db, coin, blockchair_client):
    try:
        coin_id = insert_coin_data(db, coin['name'], coin['symbol'], coin['is_scam'])
        blockchain = coin['blockchain']
        for address_info in coin['addresses']:
            address = address_info['address']
            transactions = blockchair_client.get_transactions(blockchain, address)
            if transactions:
                insert_wallet_data(db, coin_id, address, transactions)
            else:
                print(f"No transactions found for wallet {address}")
    except Exception as e:
        print(f"Error processing wallet transactions for coin {coin['name']}: {e}")
        raise

def process_coin(coin, blockchair_client):
    db = Database()
    try:
        coin_id = insert_coin_data(db, coin['name'], coin['symbol'], coin['is_scam'])
        if coin_id is None:
            print(f"Skipping processing for {coin['name']} due to failure in inserting coin data.")
            return

        print(f"Inserted/Found coin {coin['name']} with ID {coin_id}")

        latest_prices, latest_volumes = CryptoDataClient.get_latest_data(coin['symbol'])
        if latest_prices and latest_volumes:
            insert_historical_data(db, coin_id, latest_prices, 'price')
            insert_historical_data(db, coin_id, latest_volumes, 'volume')

        historical_data = CryptoDataClient.get_historical_data(coin['symbol'])
        if historical_data:
            insert_historical_data(db, coin_id, historical_data)

        general_news_articles = NewsAPIClient.get_news(coin['name'])
        print(f"General news articles fetched: {len(general_news_articles)}")
        if general_news_articles:
            insert_news_data(db, coin_id, general_news_articles)

        if coin['is_scam']:
            scam_news_articles = NewsAPIClient.get_news_for_scam(coin['name'])
            print(f"Scam news articles fetched: {len(scam_news_articles)}")
            if scam_news_articles:
                insert_news_data(db, coin_id, scam_news_articles)

        reddit_posts = RedditClient().get_reddit_posts(f"{coin['name']} scam",
                                                       datetime.now(timezone.utc) - timedelta(days=365 * 5),
                                                       datetime.now(timezone.utc))
        print(f"Reddit posts fetched: {len(reddit_posts)}")
        if reddit_posts:
            insert_social_media_data(db, coin_id, 'Reddit', reddit_posts)

        if coin['is_scam']:
            scam_reddit_posts = RedditClient().get_reddit_posts(f"{coin['name']} scam",
                                                                datetime.now(timezone.utc) - timedelta(days=365 * 5),
                                                                datetime.now(timezone.utc))
            print(f"Scam Reddit posts fetched: {len(scam_reddit_posts)}")
            if scam_reddit_posts:
                insert_social_media_data(db, coin_id, 'Reddit', scam_reddit_posts)

        process_wallet_transactions(db, coin, blockchair_client)

    except Exception as e:
        print(f"An error occurred while processing {coin['name']}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        time.sleep(15)

def parse_csv_data(csv_content):
    transactions = []
    csv_reader = csv.DictReader(io.StringIO(csv_content))
    for row in csv_reader:
        transactions.append({
            'transaction_id': row['transaction_hash'],
            'transaction_details': row,
            'time': row['time']
        })
    return transactions

if __name__ == "__main__":
    blockchair_client = BlockchairClient()
    for coin in coins:
        try:
            process_coin(coin, blockchair_client)
        except Exception as e:
            print(f"An error occurred while processing {coin['name']}: {e}")
            import traceback
            traceback.print_exc()

