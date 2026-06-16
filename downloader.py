import time
import os
from datetime import datetime

import pandas as pd
import boto3

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# =========================
# CONFIG
# =========================

ALPACA_KEY = "PKA4A6THLEKI6QD2MQPOAO25J3"
ALPACA_SECRET = "4nj9w53vMrNKJGZsHqN7Siqy34z2Gis9TffWi2beszNU"

R2_ACCESS_KEY = "00e18b0c16ecb3395cd6f7c8e0eb3554"
R2_SECRET_KEY = "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"
R2_ENDPOINT = "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"
R2_BUCKET = "stocks-data"

SYMBOL_FILE = "symbols.txt"

START_DATE = datetime(2000, 1, 1)
END_DATE = datetime(2026, 1, 1)

BATCH_SIZE = 50          # 🔥 OPTIMAL (50–100 range)
REQUESTS_PER_MIN = 200   # Alpaca limit

# =========================
# CLIENTS
# =========================

alpaca = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

s3 = boto3.client(
    "s3",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    endpoint_url=R2_ENDPOINT
)

# =========================
# RATE LIMITER
# =========================

calls = []

def rate_limit():
    global calls
    now = time.time()

    calls = [c for c in calls if now - c < 60]

    if len(calls) >= REQUESTS_PER_MIN:
        sleep_time = 60 - (now - calls[0])
        if sleep_time > 0:
            time.sleep(sleep_time)

    calls.append(time.time())

# =========================
# LOAD SYMBOLS
# =========================

def load_symbols():
    with open(SYMBOL_FILE, "r") as f:
        return [x.strip() for x in f if x.strip()]

# =========================
# CHUNKER
# =========================

def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# =========================
# PROCESS BATCH
# =========================

def process_batch(batch, batch_id):
    try:
        rate_limit()

        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Minute,
            start=START_DATE,
            end=END_DATE
        )

        bars = alpaca.get_stock_bars(request)
        df = bars.df

        if df.empty:
            print(f"[SKIP] batch {batch_id}")
            return

        df = df.reset_index()

        file_path = f"/tmp/batch_{batch_id}.parquet"

        # 🔥 fast compression
        df.to_parquet(file_path, compression="snappy")

        # upload to R2
        s3.upload_file(
            file_path,
            R2_BUCKET,
            f"batch_{batch_id}.parquet"
        )

        os.remove(file_path)

        print(f"[OK] batch {batch_id} | symbols={len(batch)} | rows={len(df)}")

    except Exception as e:
        print(f"[ERROR] batch {batch_id}: {e}")

# =========================
# MAIN
# =========================

def main():
    symbols = load_symbols()

    print("Total symbols:", len(symbols))
    print("Batch size:", BATCH_SIZE)

    batches = list(chunk(symbols, BATCH_SIZE))

    start = time.time()

    for i, batch in enumerate(batches):
        process_batch(batch, i)

        # progress
        if i % 5 == 0:
            elapsed = time.time() - start
            print(f"Progress: {i}/{len(batches)} | elapsed {elapsed/60:.2f} min")

    print("DONE")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    main()