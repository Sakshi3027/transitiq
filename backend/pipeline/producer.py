import json
import time
import signal
import sys
from confluent_kafka import Producer
from loguru import logger
from backend.core.config import settings
from backend.pipeline.simulator import generate_transit_event


class TransitProducer:
    def __init__(self):
        self.producer = Producer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "client.id": "transitiq-producer",
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 500,
        })
        self.running = True
        self.events_sent = 0

        # Graceful shutdown on Ctrl+C
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info(f"Shutting down producer. Total events sent: {self.events_sent}")
        self.running = False

    def delivery_report(self, err, msg):
        if err:
            logger.error(f"❌ Delivery failed: {err}")
        else:
            logger.debug(f"✅ Delivered to {msg.topic()} partition [{msg.partition()}]")

    def send_event(self, event: dict):
        self.producer.produce(
            topic=settings.kafka_topic_transit_events,
            key=event["route_id"].encode("utf-8"),
            value=json.dumps(event).encode("utf-8"),
            callback=self.delivery_report,
        )
        self.producer.poll(0)  # non-blocking
        self.events_sent += 1

    def run(self, events_per_second: int = 5):
        logger.info(f"🚇 TransitIQ Producer started — {events_per_second} events/sec")
        logger.info(f"📡 Sending to topic: {settings.kafka_topic_transit_events}")

        interval = 1.0 / events_per_second

        while self.running:
            event = generate_transit_event()
            self.send_event(event)

            if self.events_sent % 50 == 0:
                logger.info(f"📊 Total events produced: {self.events_sent}")

            time.sleep(interval)

        self.producer.flush()
        logger.info("Producer flushed and stopped.")


if __name__ == "__main__":
    producer = TransitProducer()
    producer.run(events_per_second=5)