"""
RabbitMQ Consumer for the Recommendation System.

Listens on:
  - rs_feedback_queue   → individual user-interaction events (rating/fav/history)
  - rs_retrain_queue    → admin-triggered full retraining requests

Runs in a background daemon thread so it does not block the FastAPI event loop.
"""
import pika
import json
import os
import threading
from src.utils.logging_config import logger


class RSConsumer:
    """Blocking consumer that processes feedback and retrain events."""

    def __init__(self, recommender, on_retrain_request, publisher=None):
        """
        Args:
            recommender:        HybridRecommender instance for online-learning updates.
            on_retrain_request: Callable that triggers a full model retrain.
            publisher:          Optional RSPublisher for cache-invalidation messages.
        """
        self.recommender = recommender
        self.on_retrain_request = on_retrain_request
        self.publisher = publisher
        self.url = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672')
        self._thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Launch the consumer in a background daemon thread."""
        self._thread = threading.Thread(target=self._consume, daemon=True, name="rs-consumer")
        self._thread.start()
        logger.info("RSConsumer: Background thread started")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _consume(self):
        """Open a blocking connection and start consuming."""
        try:
            params = pika.URLParameters(self.url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()

            # Declare the queues we listen on (idempotent — safe even if they already exist).
            channel.queue_declare(queue='rs_feedback_queue', durable=True)
            channel.queue_declare(queue='rs_retrain_queue', durable=True)

            # Process one message at a time to avoid overwhelming the recommender.
            channel.basic_qos(prefetch_count=1)

            channel.basic_consume('rs_feedback_queue', self._handle_feedback)
            channel.basic_consume('rs_retrain_queue', self._handle_retrain)

            logger.info("RSConsumer: Listening on rs_feedback_queue, rs_retrain_queue")
            channel.start_consuming()
        except Exception as e:
            logger.error(f"RSConsumer: Fatal error in consumer thread — {e}")

    # ------------------------------------------------------------------
    # Feedback handler
    # ------------------------------------------------------------------

    def _handle_feedback(self, ch, method, _props, body):
        try:
            msg = json.loads(body)
            job_id = msg.get('jobId', '?')
            logger.info(
                f"RSConsumer: Processing FEEDBACK {job_id} — "
                f"user={msg.get('userId')}, book={msg.get('bookId')}, event={msg.get('event')}"
            )

            strength = self._compute_strength(msg)

            if self.recommender and self.recommender.online_learning:
                # Collect user IDs that are in the buffer before this interaction
                buffer_before = [i['user_id'] for i in self.recommender.interaction_buffer]

                buffer_triggered = self.recommender.add_interaction(
                    user_id=msg['userId'],
                    book_id=msg['bookId'],
                    strength=strength,
                    interaction_type=msg['event'],
                )

                # If the buffer flushed, notify Node backend for cache invalidation
                if buffer_triggered and self.publisher:
                    updated_user_ids = list(set(buffer_before + [msg['userId']]))
                    logger.info(
                        f"RSConsumer: Buffer triggered — notifying cache invalidation "
                        f"for {len(updated_user_ids)} users"
                    )
                    self.publisher.publish_incremental_update(updated_user_ids)
            else:
                logger.info(f"RSConsumer: Online learning disabled — feedback logged only ({job_id})")

            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f"RSConsumer: Feedback handler error — {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    # ------------------------------------------------------------------
    # Retrain handler
    # ------------------------------------------------------------------

    def _handle_retrain(self, ch, method, _props, body):
        try:
            msg = json.loads(body)
            job_id = msg.get('jobId', '?')
            logger.info(f"RSConsumer: Processing FULL_RETRAIN {job_id} (requested by {msg.get('requestedBy', '?')})")

            self.on_retrain_request()

            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"RSConsumer: Retrain {job_id} completed")
        except Exception as e:
            logger.error(f"RSConsumer: Retrain handler error — {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    # ------------------------------------------------------------------
    # Strength mapping (mirrors the logic in routes.py record_feedback)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_strength(msg):
        event = msg.get('event')
        if event == 'rating':
            return float(msg.get('ratingValue') or 1)
        elif event == 'favorite':
            return 5.0 if (msg.get('ratingValue') or 5) > 0 else 0.0
        elif event == 'history':
            progress = msg.get('progress') or 0
            return max(0.5, (progress / 100.0) * 5.0)
        return 1.0
