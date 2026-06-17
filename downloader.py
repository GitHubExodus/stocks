import os
import time
import tempfile
import requests
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# =========================
# ENV
# =========================

ALPACA_KEY = os.environ["ALPACA_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET"]

R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_BUCKET = os.environ["R2_BUCKET"]

SERVER_URL = os.environ["SERVER_URL"]


alpaca = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

s3 = boto3.client(
    "s3",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    endpoint_url=R2_ENDPOINT
)


# =========================
# LOG
# =========================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# =========================
# CHECK IF EXISTS IN R2
# =========================

def exists(symbol):
    try:
        s3.head_object(Bucket=R2_BUCKET, Key=f"{symbol}.parquet")
        return True
    except ClientError:
        return False


# =========================
# GET SYMBOL FROM SERVER
# =========================

def get_symbol():
    try:
        r = requests.get(f"{SERVER_URL}/next", timeout=10)
        return r.json().get("symbol")
    except:
        return None


# =========================
# REPORT STATUS
# =========================

def report(symbol, status):
    try:
        requests.post(
            f"{SERVER_URL}/done",
            json={"symbol": symbol, "status": status},
            timeout=10
        )
    except:
        pass


# =========================
# DOWNLOAD DATA
# =========================

def download(symbol):
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=datetime(2000, 1, 1),
        end=datetime(2026, 6, 12)
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
# MAIN LOOP
# =========================

while True:

    symbol = get_symbol()

    if symbol is None:
        log("QUEUE EMPTY → STOPPING WORKER")
        break

    log(f"Processing: {symbol}")

    if exists(symbol):
        log("Already exists in R2 → skip")
        report(symbol, "ok")
        continue

    success = False

    for attempt in range(5):

        try:
            log(f"Attempt {attempt+1}")

            df = download(symbol)

            if df is None:
                report(symbol, "ok")
                success = True
                break

            tmp = os.path.join(tempfile.gettempdir(), f"{symbol}.parquet")

            df.to_parquet(tmp, compression="snappy")

            upload(symbol, tmp)

            os.remove(tmp)

            report(symbol, "ok")

            log("DONE")
            success = True
            break

        except Exception as e:
            wait = 2 ** attempt
            log(f"ERROR: {e}")
            log(f"Retrying in {wait}s")
            time.sleep(wait)

    if not success:
        log("FAILED after retries")
        report(symbol, "fail")


log("WORKER EXITED")