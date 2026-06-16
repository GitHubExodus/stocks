import os
import time
import tempfile
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ==================================
# CONFIG
# ==================================

ALPACA_KEY = "PKA4A6THLEKI6QD2MQPOAO25J3"
ALPACA_SECRET = "4nj9w53vMrNKJGZsHqN7Siqy34z2Gis9TffWi2beszNU"

R2_ACCESS_KEY = "00e18b0c16ecb3395cd6f7c8e0eb3554"
R2_SECRET_KEY = "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"
R2_ENDPOINT = "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"
R2_BUCKET = "stocks-data"

SYMBOL_FILE = "symbols.txt"

START_DATE = datetime(2000, 1, 1)
END_DATE = datetime(2026, 6, 12)

# ===== POD SETTINGS =====
POD_INDEX = 3        # <-- change per pod (0–9)
TOTAL_PODS = 10

REQUESTS_PER_MIN = 20
SECONDS_PER_REQUEST = 60 / REQUESTS_PER_MIN


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
# LOAD + SHARD
# =========================

with open(SYMBOL_FILE) as f:
    all_symbols = [x.strip() for x in f if x.strip()]

# 🔥 THIS IS THE SHARDING LOGIC
symbols = all_symbols[POD_INDEX::TOTAL_PODS]

total = len(symbols)

print(f"TOTAL GLOBAL: {len(all_symbols)}")
print(f"POD {POD_INDEX}/{TOTAL_PODS}")
print(f"THIS POD SYMBOLS: {total}")


# =========================
# RATE LIMIT
# =========================

last_request = 0

def wait_rate_limit():
    global last_request

    elapsed = time.time() - last_request

    if elapsed < SECONDS_PER_REQUEST:
        time.sleep(SECONDS_PER_REQUEST - elapsed)

    last_request = time.time()


# =========================
# CHECK EXISTS
# =========================

def exists(symbol):
    try:
        s3.head_object(Bucket=R2_BUCKET, Key=f"{symbol}.parquet")
        return True
    except ClientError:
        return False


# =========================
# DOWNLOAD
# =========================

def download(symbol):

    wait_rate_limit()

    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=START_DATE,
        end=END_DATE
    )

    bars = alpaca.get_stock_bars(req)

    df = bars.df

    if df.empty:
        return None

    return df.reset_index()


# =========================
# UPLOAD
# =========================

def upload(symbol, path):
    s3.upload_file(path, R2_BUCKET, f"{symbol}.parquet")


# =========================
# MAIN
# =========================

for i, symbol in enumerate(symbols, start=1):

    print("\n" + "=" * 50)
    print(f"[POD {POD_INDEX}] {i}/{total} -> {symbol}")

    if exists(symbol):
        print("SKIP (already in R2)")
        continue

    try:
        print("Downloading...")

        df = download(symbol)

        if df is None:
            print("NO DATA")
            continue

        tmp = os.path.join(
            tempfile.gettempdir(),
            f"{symbol}.parquet"
        )

        print("Writing parquet...")

        df.to_parquet(tmp, compression="snappy")

        print("Uploading...")

        upload(symbol, tmp)

        os.remove(tmp)

        print("DONE")

    except Exception as e:
        print(f"ERROR: {e}")

print("\nPOD COMPLETE")