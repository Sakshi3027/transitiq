import json
import signal
from confluent_kafka import Consumer, KafkaError, KafkaException
from loguru import logger
from backend.core.config import settings


class TransitConsumer:
    def __init__(self):
        self.consumer = Consumer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": settings.kafka_group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 1000,
        })
        self.running = True
        self.events_processed = 0

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info(f"Shutting down consumer. Events processed: {self.events_processed}")
        self.running = False

    def process_event(self, event: dict):
        """
        This is where each event gets processed.
        In Phase 3 we'll plug in the ML model here.
        For now we just log and store.
        """
        route = event.get("route_id")
        delay = event.get("delay_minutes", 0)
        weather = event.get("weather_condition")
        peak = event.get("is_peak_hour")

        if delay > 15:
            logger.warning(
                f"🚨 HIGH DELAY | Route: {route} | "
                f"Delay: {delay:.1f} min | Weather: {weather} | Peak: {peak}"
            )
        else:
            logger.info(
                f"🚇 Event | Route: {route} | "
                f"Delay: {delay:.1f} min | Weather: {weather}"
            )

        self.events_processed += 1

        if self.events_processed % 100 == 0:
            logger.info(f"📊 Total events consumed: {self.events_processed}")

    def run(self):
        self.consumer.subscribe([settings.kafka_topic_transit_events])
        logger.info(f"🎧 TransitIQ Consumer started")
        logger.info(f"📡 Subscribed to: {settings.kafka_topic_transit_events}")

        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug("Reached end of partition")
                    else:
                        raise KafkaException(msg.error())
                    continue

                event = json.loads(msg.value().decode("utf-8"))
                self.process_event(event)

        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.consumer.close()
            logger.info("Consumer closed.")


if __name__ == "__main__":
    consumer = TransitConsumer()
    consumer.run()