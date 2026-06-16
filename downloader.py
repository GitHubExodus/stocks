import time
import os
from datetime import datetime

import pandas as pd
import boto3
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# =========================
# CONFIG (DO NOT HARDCODE IN REAL USE)
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

REQUESTS_PER_MIN = 200

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
# RATE LIMIT
# =========================

calls = []

def rate_limit():
    global calls
    now = time.time()

    calls = [c for c in calls if now - c < 60]

    if len(calls) >= REQUESTS_PER_MIN:
        sleep_time = 60 - (now - calls[0])
        time.sleep(max(0, sleep_time))

    calls.append(time.time())

# =========================
# SYMBOLS
# =========================

def load_symbols():
    with open(SYMBOL_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]

# =========================
# DOWNLOAD + UPLOAD
# =========================

def process_symbol(symbol):
    try:
        rate_limit()

        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=START_DATE,
            end=END_DATE
        )

        bars = alpaca.get_stock_bars(request)
        df = bars.df

        if df.empty:
            print(f"SKIP {symbol}")
            return

        df = df.reset_index()

        file_path = f"/tmp/{symbol}.parquet"
        df.to_parquet(file_path, compression="snappy")

        s3.upload_file(
            file_path,
            R2_BUCKET,
            f"{symbol}.parquet"
        )

        os.remove(file_path)

        print(f"OK {symbol}")

    except Exception as e:
        print(f"ERROR {symbol}: {e}")

# =========================
# MAIN LOOP
# =========================

def main():
    symbols = load_symbols()
    print("Total symbols:", len(symbols))

    for i, symbol in enumerate(symbols):
        process_symbol(symbol)

        if i % 100 == 0:
            print(f"Progress: {i}/{len(symbols)}")

if __name__ == "__main__":
    main()