"""
RabbitMQ Publisher for the Recommendation System.

Replaces the synchronous HTTP callbacks in ``callback.py`` with durable
RabbitMQ messages.  The Node.js ``cache-invalidation.worker.js`` consumes
these messages and invalidates the appropriate Redis cache keys.
"""
import pika
import json
import os
from src.utils.logging_config import logger


class RSPublisher:
    """Publishes cache-invalidation events to RabbitMQ."""

    QUEUE = 'cache_invalidation_queue'

    def __init__(self):
        self.url = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672')
        self._connection = None
        self._channel = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Open a blocking connection and declare the target queue."""
        try:
            params = pika.URLParameters(self.url)
            self._connection = pika.BlockingConnection(params)
            self._channel = self._connection.channel()
            self._channel.queue_declare(queue=self.QUEUE, durable=True)
            logger.info(f"RSPublisher: Connected — queue '{self.QUEUE}' asserted")
        except Exception as e:
            logger.error(f"RSPublisher: Failed to connect — {e}")
            self._connection = None
            self._channel = None

    def disconnect(self):
        """Gracefully close the connection."""
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
                logger.info("RSPublisher: Disconnected")
        except Exception as e:
            logger.error(f"RSPublisher: Error during disconnect — {e}")
        finally:
            self._connection = None
            self._channel = None

    # ------------------------------------------------------------------
    # Public publish helpers
    # ------------------------------------------------------------------

    def publish_retrain_complete(self, model_key=None):
        """Notify Node backend that a full retrain has finished."""
        self._publish({
            'type': 'RETRAIN_COMPLETE',
            'modelKey': model_key,
        })

    def publish_incremental_update(self, user_ids):
        """Notify Node backend that specific users' profiles were updated."""
        self._publish({
            'type': 'INCREMENTAL_UPDATE',
            'userIds': user_ids,
        })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_connected(self):
        """Reconnect lazily if the connection was lost."""
        if self._channel and self._connection and self._connection.is_open:
            return True
        logger.warn("RSPublisher: Connection lost — attempting reconnect")
        self.connect()
        return self._channel is not None

    def _publish(self, payload):
        if not self._ensure_connected():
            logger.warn("RSPublisher: Not connected — dropping message", )
            return

        try:
            self._channel.basic_publish(
                exchange='',
                routing_key=self.QUEUE,
                body=json.dumps(payload),
                properties=pika.BasicProperties(delivery_mode=2),  # persistent
            )
            logger.info(f"RSPublisher: Published {payload.get('type')} to {self.QUEUE}")
        except Exception as e:
            logger.error(f"RSPublisher: Failed to publish — {e}")
            # Mark connection as dead so next call tries to reconnect
            self._channel = None
            self._connection = None
