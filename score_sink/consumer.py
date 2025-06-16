import asyncio, os, orjson, asyncpg
from aiokafka import AIOKafkaConsumer
import logging, sys


KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "scores")

DSN = ("postgresql://{user}:{pwd}@{host}/{db}".format(user=os.getenv("PG_USER","fraud"),
        pwd=os.getenv("PG_PASS","fraud"), host=os.getenv("PG_HOST","db"), db=os.getenv("PG_DB"  ,"fraud"))
)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info("consumer.py STARTED")
logging.info("Kafka bootstrap=%s  topic=%s", KAFKA_BOOTSTRAP, TOPIC)
logging.info("Postgres DSN=%s", DSN.split('@')[-1])

async def main():
    logging.info("Creating PG pool…")
    pg = await asyncpg.create_pool(dsn=DSN, min_size=1, max_size=5)

    logging.info("Starting Kafka consumer…")
    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="score_sink_debug",
        auto_offset_reset="earliest",
        value_deserializer=lambda b: orjson.loads(b),
    )
    await consumer.start()
    logging.info("Kafka SUBSCRIPTION: %s", consumer.subscription())

    try:
        async for msg in consumer:
            raw = msg.value
            if isinstance(raw, dict):
                tx = raw
            elif isinstance(raw, list):
                if len(raw) == 1 and isinstance(raw[0], dict):
                    tx = raw[0]
                elif len(raw) >= 3:
                    tx = {"transaction_id": raw[0], "score": raw[1], "fraud_flag": raw[2]}
                else:
                    logging.warning("Unknown list format: %s", raw)
                    continue
            else:
                logging.warning("Unknown message type: %s", type(raw))
                continue

            logging.info("📩 normalised msg: %s", tx)
            await pg.execute(
                """
                INSERT INTO fraud_scores(transaction_id, score, fraud_flag)
                VALUES($1,$2,$3) ON CONFLICT DO NOTHING
                """,
                tx["transaction_id"], tx["score"], tx["fraud_flag"]
            )
            logging.info("inserted tx=%s", tx["transaction_id"])
    finally:
        await consumer.stop()
        await pg.close()

if __name__ == "__main__":
    asyncio.run(main())