# worker.py
import os
import sys

sys.path.append(os.path.dirname(__file__))

import models.predict_new_coin

from dramatiq import set_broker
from dramatiq.brokers.redis import RedisBroker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

redis_broker = RedisBroker(url="redis://localhost:6379/0")
set_broker(redis_broker)

if __name__ == "__main__":
    from dramatiq.cli import main
    logger.info("Starting Dramatiq worker...")
    main()
