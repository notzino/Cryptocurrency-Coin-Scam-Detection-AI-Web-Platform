import os

twitter_consumer_key = os.environ.get('TWITTER_CONSUMER_KEY', '')
twitter_consumer_secret = os.environ.get('TWITTER_CONSUMER_SECRET', '')
twitter_access_token = os.environ.get('TWITTER_ACCESS_TOKEN', '')
twitter_access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET', '')
twitter_bearer_token = os.environ.get('TWITTER_BEARER_TOKEN', '')
currents_api_key = os.getenv('CURRENTS_API_KEY', '')
coinmarketcap_api_key = os.getenv('COINMARKETCAP_API_KEY', '')
reddit_client_id = os.getenv('REDDIT_CLIENT_ID', '')
reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
reddit_user_agent = os.getenv('REDDIT_USER_AGENT', '')
blockchain_exchange_api_key = os.getenv('BLOCKCHAIN_EXCHANGE_API_KEY', '')
blockchain_exchange_api_secret = os.getenv('BLOCKCHAIN_EXCHANGE_API_SECRET', '')
thenewsapi_token = os.getenv('THENEWSAPI_TOKEN', '') #https://www.thenewsapi.com/
openai_organization_id = os.getenv('OPENAI_ORGANIZATION_ID', '')
openai_api_key = os.getenv('OPENAI_API_KEY', '')
blockchair_api_key = os.getenv('BLOCKCHAIR_API_KEY', '')