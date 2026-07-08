# ==========================================================
# IMPORTS
# ==========================================================

import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from numba import njit, prange


# ==========================================================
# IMPORTS
# ==========================================================

import gc
import os
import random
from pathlib import Path
from time import perf_counter

import boto3
import botocore

import numpy as np
import pandas as pd

from numba import njit, prange

# ==========================================================
# CLOUDFLARE R2 CONFIG
# ==========================================================

R2_ACCESS_KEY = "00e18b0c16ecb3395cd6f7c8e0eb3554"
R2_SECRET_KEY = "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"
R2_ENDPOINT = "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"
R2_BUCKET = "stocks-data"

r2 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
)

# ==========================================================
# LOCAL FILES
# ==========================================================

WORK_DIR = Path("work")
WORK_DIR.mkdir(exist_ok=True)

SYMBOLS_FILE = WORK_DIR / "symbols.txt"

LOCAL_PARQUET = WORK_DIR / "stock.parquet"

INPUT_OUTPUT = WORK_DIR / "indicator_features.parquet"

SIGNAL_OUTPUT = WORK_DIR / "signal_labels.parquet"

# ==========================================================
# R2 PATHS
# ==========================================================

SYMBOL_LIST_KEY = "misc/symbols.txt"

INPUT_FOLDER = "input"

OUTPUT_FOLDER = "output"

# ==========================================================
# PERIODS
# ==========================================================

PERIODS = np.array(
    [
        3,
        5,
        9,
        14,
        21,
        50,
        100,
        200,
        500,
        1000,
    ],
    dtype=np.int32,
)

PIVOT_PERIODS = np.array(
    [
        1,
        2,
        4,
        8,
    ],
    dtype=np.int32,
)

# ==========================================================
# DATATYPES
# ==========================================================

FLOAT = np.float32
INT = np.int32

# ----------------------------------------------------------
# FEATURE SETTINGS
# ----------------------------------------------------------

SAVE_PARQUET = True
OUTPUT_FILE = "training_features.parquet"









































# ==========================================================
# R2 HELPERS
# ==========================================================

def download_file(key, destination):

    r2.download_file(
        R2_BUCKET,
        key,
        str(destination),
    )


def upload_file(source, key):

    r2.upload_file(
        str(source),
        R2_BUCKET,
        key,
    )


def object_exists(key):

    try:

        r2.head_object(
            Bucket=R2_BUCKET,
            Key=key,
        )

        return True

    except botocore.exceptions.ClientError:

        return False


def download_symbols():

    print("Downloading symbols list...")

    download_file(
        SYMBOL_LIST_KEY,
        SYMBOLS_FILE,
    )

    with open(
        SYMBOLS_FILE,
        "r",
        encoding="utf8",
    ) as f:

        symbols = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return symbols


def upload_symbols(symbols):

    with open(
        SYMBOLS_FILE,
        "w",
        encoding="utf8",
    ) as f:

        f.write("\n".join(symbols))

    upload_file(
        SYMBOLS_FILE,
        SYMBOL_LIST_KEY,
    )


def download_stock(symbol):

    key = f"{symbol}.parquet"

    if not object_exists(key):

        return False

    print(f"Downloading {symbol}...")

    download_file(
        key,
        LOCAL_PARQUET,
    )

    return True


def upload_results(symbol):

    input_key = (
        f"{INPUT_FOLDER}/{symbol}.parquet"
    )

    output_key = (
        f"{OUTPUT_FOLDER}/{symbol}.parquet"
    )

    print("Uploading input features...")

    upload_file(
        INPUT_OUTPUT,
        input_key,
    )

    print("Uploading signal labels...")

    upload_file(
        SIGNAL_OUTPUT,
        output_key,
    )









# ==========================================================
# GET RANDOM STOCK
# ==========================================================

print("Loading symbol list...")

symbols = download_symbols()

if len(symbols) == 0:

    raise RuntimeError(
        "symbols.txt is empty."
    )

random.shuffle(symbols)

SYMBOL = None

while symbols:

    candidate = symbols.pop()

    print(f"Trying {candidate}...")

    if download_stock(candidate):

        SYMBOL = candidate

        upload_symbols(symbols)

        break

    print(f"{candidate} not found.")

if SYMBOL is None:

    raise RuntimeError(
        "No valid stock parquet files were found."
    )

print()
print(f"Selected Stock : {SYMBOL}")
print()













# ==========================================================
# LOAD DATA
# ==========================================================

print(f"Loading {SYMBOL}...")

df = pd.read_parquet(
    LOCAL_PARQUET,
)

open_ = df["open"].to_numpy(dtype=FLOAT)
high = df["high"].to_numpy(dtype=FLOAT)
low = df["low"].to_numpy(dtype=FLOAT)
close = df["close"].to_numpy(dtype=FLOAT)
volume = df["volume"].to_numpy(dtype=FLOAT)

vwap_price = df["vwap"].to_numpy(dtype=FLOAT)

# ----------------------------------------------------------
# Optional timestamp (saved with signal file)
# ----------------------------------------------------------

if "timestamp" in df.columns:

    timestamp = df["timestamp"].to_numpy()

elif "date" in df.columns:

    timestamp = df["date"].to_numpy()

elif "time" in df.columns:

    timestamp = df["time"].to_numpy()

ROWS = len(close)

del df
gc.collect()

print(f"Rows Loaded : {ROWS:,}")

















# ==========================================================
# TIMER
# ==========================================================

class Timer:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = perf_counter() - self.start
        print(f"{self.name:<30}{elapsed:.3f} sec")



















# ==========================================================
# FEATURE ENGINE
# ==========================================================

class FeatureEngine:

    def __init__(self, rows):

        self.rows = rows

        self.columns = []

        self.names = []

    def add(self, name, values):

        values = np.asarray(values, dtype=np.float32)

        if values.ndim == 1:

            values = values.reshape(-1, 1)

        self.columns.append(values)

        if values.shape[1] == 1:

            self.names.append(name)

        else:

            for i in range(values.shape[1]):

                self.names.append(f"{name}_{i}")

    def build(self):

        return np.ascontiguousarray(
            np.column_stack(self.columns),
            dtype=np.float32,
        )
    














# ==========================================================
# INDICATOR ENGINE
# ==========================================================

class Indicator:

    def __init__(self, values, periods):

        self.values = np.ascontiguousarray(values)

        self.periods = periods

        self.lookup = {
            int(period): index
            for index, period in enumerate(periods)
        }

    def get(self, period):

        return self.values[:, self.lookup[int(period)]]
    


























































# ==========================================================
# EMA ENGINE
# ==========================================================

@njit(cache=True)
def _ema_single(source, period):
    """
    Computes a single EMA.

    Parameters
    ----------
    source : float32[:]
    period : int

    Returns
    -------
    ema : float32[:]
    """

    rows = source.shape[0]

    ema = np.empty(rows, dtype=np.float32)

    alpha = np.float32(2.0 / (period + 1.0))

    ema[0] = source[0]

    for i in range(1, rows):

        ema[i] = ema[i - 1] + alpha * (source[i] - ema[i - 1])

    return ema


@njit(parallel=True, cache=True)
def _ema_all(source, periods):
    """
    Computes EMAs for every period.

    Returns
    -------
    (rows, periods)
    """

    rows = source.shape[0]
    total_periods = periods.shape[0]

    output = np.empty((rows, total_periods), dtype=np.float32)

    for p in prange(total_periods):

        output[:, p] = _ema_single(source, periods[p])

    return output


def build_ema(source, periods=PERIODS):
    """
    Build all EMA periods.

    Returns
    -------
    Indicator
    """

    with Timer("Building EMA"):

        values = _ema_all(source, periods)

    return Indicator(values, periods)



















# ==========================================================
# TEST EMA
# ==========================================================

ema = build_ema(close)

print()

print("EMA Shape")
print(ema.values.shape)

print()

print("Periods")
print(ema.periods)

print()

print("EMA 21")
print(ema.get(21)[:10])


























































# ==========================================================
# FEATURE MATRIX
# ==========================================================

class FeatureMatrix:

    def __init__(self):

        self.data = {}
        self.count = 0

    def add(self, name, values):

        values = np.asarray(values, dtype=np.float32)

        if values.ndim == 1:

            self.data[name] = values
            self.count += 1

        else:

            cols = values.shape[1]

            for i in range(cols):

                self.data[f"{name}_{i}"] = values[:, i]

                self.count += 1

    def dataframe(self):

        return pd.DataFrame(self.data)


features = FeatureMatrix()




















# ==========================================================
# EMA PERCENTAGE DISTANCE
# ==========================================================

@njit(parallel=True, cache=True)
def ema_percent_distance(close, ema):

    rows = ema.shape[0]
    cols = ema.shape[1]

    output = np.empty((rows, cols), dtype=np.float32)

    for j in prange(cols):

        for i in range(rows):

            e = ema[i, j]

            if e == 0:

                output[i, j] = 0

            else:

                output[i, j] = (
                    (close[i] - e) / e
                ) * 100.0

    return output


with Timer("EMA % Distance"):

    ema_pct = ema_percent_distance(
        close,
        ema.values,
    )

features.add("ema_pct", ema_pct)













print()

print(ema_pct.shape)

print()

print(ema_pct[:5])

























# ==========================================================
# EMA CROSS EVENTS
# ==========================================================

@njit(cache=True)
def cross_events(a, b):
    """
    Returns

        1  -> Cross Up
       -1  -> Cross Down
        0  -> No Cross
    """

    rows = a.shape[0]

    out = np.zeros(rows, dtype=np.int8)

    for i in range(1, rows):

        prev = a[i - 1] - b[i - 1]
        curr = a[i] - b[i]

        if prev <= 0.0 and curr > 0.0:

            out[i] = 1

        elif prev >= 0.0 and curr < 0.0:

            out[i] = -1

    return out


with Timer("EMA Cross Events"):

    # --------------------------------------------------
    # Close crosses each EMA
    # --------------------------------------------------

    for period in ema.periods:

        features.add(
            f"close_cross_ema_{period}",
            cross_events(
                close,
                ema.get(period),
            ),
        )

    # --------------------------------------------------
    # EMA crosses EMA
    # --------------------------------------------------

    periods = ema.periods

    for i in range(len(periods)):

        for j in range(i + 1, len(periods)):

            p1 = periods[i]
            p2 = periods[j]

            features.add(
                f"ema_{p1}_cross_ema_{p2}",
                cross_events(
                    ema.get(p1),
                    ema.get(p2),
                ),
            )

print(f"Total Features: {features.count}")























# ==========================================================
# BAR COUNT ENGINE
# ==========================================================

@njit(cache=True)
def bars_since_event(events):
    """
    Parameters
    ----------
    events : int8[:]

    Returns
    -------
    int32[:]

    Number of bars since the most recent event.
    """

    rows = events.shape[0]

    out = np.empty(rows, dtype=np.int32)

    count = 1_000_000

    for i in range(rows):

        if events[i] != 0:

            count = 0

        else:

            count += 1

        out[i] = count

    return out

















# ==========================================================
# EMA BAR COUNTS
# ==========================================================

with Timer("EMA Bar Counts"):

    # --------------------------------------------------
    # Close vs EMA
    # --------------------------------------------------

    for period in ema.periods:

        events = cross_events(
            close,
            ema.get(period),
        )

        features.add(
            f"bars_close_cross_ema_{period}",
            bars_since_event(events),
        )

    # --------------------------------------------------
    # EMA vs EMA
    # --------------------------------------------------

    periods = ema.periods

    for i in range(len(periods)):

        for j in range(i + 1, len(periods)):

            p1 = periods[i]
            p2 = periods[j]

            events = cross_events(
                ema.get(p1),
                ema.get(p2),
            )

            features.add(
                f"bars_ema_{p1}_cross_ema_{p2}",
                bars_since_event(events),
            )

print(f"Total Features: {features.count}")







































































# ==========================================================
# DEMA ENGINE
# ==========================================================

@njit(parallel=True, cache=True)
def _dema_all(source, periods):

    rows = source.shape[0]
    total_periods = periods.shape[0]

    output = np.empty((rows, total_periods), dtype=np.float32)

    for p in prange(total_periods):

        ema1 = _ema_single(source, periods[p])

        ema2 = _ema_single(ema1, periods[p])

        output[:, p] = 2.0 * ema1 - ema2

    return output


def build_dema(source, periods=PERIODS):

    with Timer("Building DEMA"):

        values = _dema_all(source, periods)

    return Indicator(values, periods)












dema = build_dema(close)

print("DEMA Shape:", dema.values.shape)
print("DEMA 21:", dema.get(21)[:10])




















# ==========================================================
# DEMA PERCENTAGE DISTANCE
# ==========================================================

with Timer("DEMA % Distance"):

    dema_pct = ema_percent_distance(
        close,
        dema.values,
    )

features.add(
    "dema_pct",
    dema_pct,
)

print(f"Total Features: {features.count}")








# ==========================================================
# DEMA CROSS EVENTS
# ==========================================================

with Timer("DEMA Cross Events"):

    # --------------------------------------------------
    # Close crosses each DEMA
    # --------------------------------------------------

    for period in dema.periods:

        features.add(
            f"close_cross_dema_{period}",
            cross_events(
                close,
                dema.get(period),
            ),
        )

    # --------------------------------------------------
    # DEMA crosses DEMA
    # --------------------------------------------------

    periods = dema.periods

    for i in range(len(periods)):

        for j in range(i + 1, len(periods)):

            p1 = periods[i]
            p2 = periods[j]

            features.add(
                f"dema_{p1}_cross_dema_{p2}",
                cross_events(
                    dema.get(p1),
                    dema.get(p2),
                ),
            )

print(f"Total Features: {features.count}")















# ==========================================================
# DEMA BAR COUNTS
# ==========================================================

with Timer("DEMA Bar Counts"):

    # --------------------------------------------------
    # Close vs DEMA
    # --------------------------------------------------

    for period in dema.periods:

        events = cross_events(
            close,
            dema.get(period),
        )

        features.add(
            f"bars_close_cross_dema_{period}",
            bars_since_event(events),
        )

    # --------------------------------------------------
    # DEMA vs DEMA
    # --------------------------------------------------

    periods = dema.periods

    for i in range(len(periods)):

        for j in range(i + 1, len(periods)):

            p1 = periods[i]
            p2 = periods[j]

            events = cross_events(
                dema.get(p1),
                dema.get(p2),
            )

            features.add(
                f"bars_dema_{p1}_cross_dema_{p2}",
                bars_since_event(events),
            )

print(f"Total Features: {features.count}")

































































# ==========================================================
# VWAP ENGINE
# ==========================================================

def build_vwap(vwap):

    with Timer("Building VWAP"):

        values = vwap.reshape(-1, 1)

    return Indicator(
        values,
        np.array([0], dtype=np.int32),
    )


vwap = build_vwap(vwap_price)

print("VWAP Shape:", vwap.values.shape)
print("VWAP:", vwap.get(0)[:10])














# ==========================================================
# VWAP PERCENTAGE DISTANCE
# ==========================================================

with Timer("VWAP % Distance"):

    vwap_pct = ema_percent_distance(
        close,
        vwap.values,
    )

features.add(
    "vwap_pct",
    vwap_pct,
)

print(f"Total Features: {features.count}")













# ==========================================================
# CLOSE CROSS VWAP
# ==========================================================

with Timer("VWAP Cross Events"):

    events = cross_events(
        close,
        vwap.get(0),
    )

    features.add(
        "close_cross_vwap",
        events,
    )

print(f"Total Features: {features.count}")










# ==========================================================
# VWAP BAR COUNT
# ==========================================================

with Timer("VWAP Bar Count"):

    features.add(
        "bars_close_cross_vwap",
        bars_since_event(events),
    )

print(f"Total Features: {features.count}")














































































# ==========================================================
# RSI ENGINE
# ==========================================================

@njit(cache=True)
def _rsi_single(source, period):

    rows = source.shape[0]

    rsi = np.empty(rows, dtype=np.float32)

    gain = np.float32(0.0)
    loss = np.float32(0.0)

    rsi[0] = 50.0

    # Initial average
    end = min(period + 1, rows)

    for i in range(1, end):

        diff = source[i] - source[i - 1]

        if diff > 0:

            gain += diff

        else:

            loss -= diff

    gain /= period
    loss /= period

    if loss == 0.0:

        rsi[period] = 100.0

    else:

        rs = gain / loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    alpha = np.float32(1.0 / period)

    for i in range(period + 1, rows):

        diff = source[i] - source[i - 1]

        up = diff if diff > 0 else 0.0
        down = -diff if diff < 0 else 0.0

        gain = gain + alpha * (up - gain)
        loss = loss + alpha * (down - loss)

        if loss == 0.0:

            rsi[i] = 100.0

        else:

            rs = gain / loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period):

        rsi[i] = 50.0

    return rsi


@njit(parallel=True, cache=True)
def _rsi_all(source, periods):

    rows = source.shape[0]

    cols = periods.shape[0]

    output = np.empty((rows, cols), dtype=np.float32)

    for p in prange(cols):

        output[:, p] = _rsi_single(
            source,
            periods[p],
        )

    return output


def build_rsi(source, periods=PERIODS):

    with Timer("Building RSI"):

        values = _rsi_all(
            source,
            periods,
        )

    return Indicator(
        values,
        periods,
    )



























# ==========================================================
# BUILD RSI
# ==========================================================

rsi = build_rsi(close)

print("RSI Shape:", rsi.values.shape)
print("RSI 14:", rsi.get(14)[:20])














# ==========================================================
# RSI FEATURES
# ==========================================================

with Timer("RSI Features"):

    features.add(
        "rsi",
        rsi.values,
    )

print(f"Total Features: {features.count}")


































































































# ==========================================================
# ROC ENGINE
# ==========================================================

@njit(cache=True)
def _roc_single(source, period):

    rows = source.shape[0]

    roc = np.empty(rows, dtype=np.float32)

    for i in range(period):

        roc[i] = 0.0

    for i in range(period, rows):

        prev = source[i - period]

        if prev == 0.0:

            roc[i] = 0.0

        else:

            roc[i] = (
                (source[i] - prev)
                / prev
            ) * 100.0

    return roc


@njit(parallel=True, cache=True)
def _roc_all(source, periods):

    rows = source.shape[0]

    cols = periods.shape[0]

    output = np.empty((rows, cols), dtype=np.float32)

    for p in prange(cols):

        output[:, p] = _roc_single(
            source,
            periods[p],
        )

    return output


def build_roc(source, periods=PERIODS):

    with Timer("Building ROC"):

        values = _roc_all(
            source,
            periods,
        )

    return Indicator(
        values,
        periods,
    )
















# ==========================================================
# BUILD ROC
# ==========================================================

roc = build_roc(close)

print("ROC Shape:", roc.values.shape)
print("ROC 14:", roc.get(14)[:20])













# ==========================================================
# ROC FEATURES
# ==========================================================

with Timer("ROC Features"):

    features.add(
        "roc",
        roc.values,
    )

print(f"Total Features: {features.count}")














































# ==========================================================
# STD ENGINE
# ==========================================================

@njit(cache=True)
def _std_single(source, period):

    rows = source.shape[0]

    out = np.empty(rows, dtype=np.float32)

    for i in range(period - 1):

        out[i] = 0.0

    for i in range(period - 1, rows):

        start = i - period + 1

        mean = 0.0

        for j in range(start, i + 1):

            mean += source[j]

        mean /= period

        var = 0.0

        for j in range(start, i + 1):

            diff = source[j] - mean

            var += diff * diff

        out[i] = np.sqrt(var / period)

    return out


@njit(parallel=True, cache=True)
def _std_all(source, periods):

    rows = source.shape[0]

    cols = periods.shape[0]

    output = np.empty((rows, cols), dtype=np.float32)

    for p in prange(cols):

        output[:, p] = _std_single(
            source,
            periods[p],
        )

    return output


def build_std(source, periods=PERIODS):

    with Timer("Building STD"):

        values = _std_all(
            source,
            periods,
        )

    return Indicator(
        values,
        periods,
    )





















# ==========================================================
# BUILD STD
# ==========================================================

std = build_std(close)

print("STD Shape:", std.values.shape)
print("STD 14:", std.get(14)[:20])






















# ==========================================================
# STD FEATURES
# ==========================================================

with Timer("STD Features"):

    features.add(
        "std",
        std.values,
    )

print(f"Total Features: {features.count}")





































































# ==========================================================
# ATR ENGINE
# ==========================================================

@njit(cache=True)
def _atr_single(high, low, close, period):

    rows = close.shape[0]

    tr = np.empty(rows, dtype=np.float32)

    tr[0] = high[0] - low[0]

    for i in range(1, rows):

        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])

        tr[i] = max(hl, hc, lc)

    atr = np.empty(rows, dtype=np.float32)

    for i in range(period):

        atr[i] = 0.0

    avg = 0.0

    for i in range(period):

        avg += tr[i]

    avg /= period

    atr[period] = avg

    alpha = np.float32(1.0 / period)

    for i in range(period + 1, rows):

        avg = avg + alpha * (tr[i] - avg)

        atr[i] = avg

    return atr


@njit(parallel=True, cache=True)
def _atr_all(high, low, close, periods):

    rows = close.shape[0]

    cols = periods.shape[0]

    output = np.empty((rows, cols), dtype=np.float32)

    for p in prange(cols):

        output[:, p] = _atr_single(
            high,
            low,
            close,
            periods[p],
        )

    return output


def build_atr(high, low, close, periods=PERIODS):

    with Timer("Building ATR"):

        values = _atr_all(
            high,
            low,
            close,
            periods,
        )

    return Indicator(
        values,
        periods,
    )














# ==========================================================
# BUILD ATR
# ==========================================================

atr = build_atr(
    high,
    low,
    close,
)

print("ATR Shape:", atr.values.shape)
print("ATR 14:", atr.get(14)[:20])

















# ==========================================================
# ATR FEATURES
# ==========================================================

with Timer("ATR Features"):

    features.add(
        "atr",
        atr.values,
    )

print(f"Total Features: {features.count}")


























































# ==========================================================
# ADX ENGINE
# ==========================================================

@njit(cache=True)
def _adx_single(high, low, close, period):

    rows = close.shape[0]

    tr = np.empty(rows, dtype=np.float32)
    plus_dm = np.empty(rows, dtype=np.float32)
    minus_dm = np.empty(rows, dtype=np.float32)

    tr[0] = 0.0
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, rows):

        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]

        if up > down and up > 0:
            plus_dm[i] = up
        else:
            plus_dm[i] = 0.0

        if down > up and down > 0:
            minus_dm[i] = down
        else:
            minus_dm[i] = 0.0

        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])

        tr[i] = max(hl, hc, lc)

    out = np.empty(rows, dtype=np.float32)

    for i in range(period):
        out[i] = 0.0

    tr_s = 0.0
    plus_s = 0.0
    minus_s = 0.0

    for i in range(1, period + 1):

        tr_s += tr[i]
        plus_s += plus_dm[i]
        minus_s += minus_dm[i]

    alpha = np.float32(1.0 / period)

    dx = np.empty(rows, dtype=np.float32)

    for i in range(period):

        dx[i] = 0.0

    for i in range(period, rows):

        if i > period:

            tr_s += alpha * (tr[i] - tr_s)
            plus_s += alpha * (plus_dm[i] - plus_s)
            minus_s += alpha * (minus_dm[i] - minus_s)

        if tr_s == 0.0:

            dx[i] = 0.0

        else:

            plus_di = 100.0 * plus_s / tr_s
            minus_di = 100.0 * minus_s / tr_s

            denom = plus_di + minus_di

            if denom == 0.0:

                dx[i] = 0.0

            else:

                dx[i] = (
                    abs(plus_di - minus_di)
                    / denom
                ) * 100.0

    adx = np.empty(rows, dtype=np.float32)

    for i in range(period):

        adx[i] = 0.0

    avg = dx[period]

    adx[period] = avg

    for i in range(period + 1, rows):

        avg += alpha * (dx[i] - avg)

        adx[i] = avg

    return adx


@njit(parallel=True, cache=True)
def _adx_all(high, low, close, periods):

    rows = close.shape[0]

    cols = periods.shape[0]

    output = np.empty((rows, cols), dtype=np.float32)

    for p in prange(cols):

        output[:, p] = _adx_single(
            high,
            low,
            close,
            periods[p],
        )

    return output


def build_adx(high, low, close, periods=PERIODS):

    with Timer("Building ADX"):

        values = _adx_all(
            high,
            low,
            close,
            periods,
        )

    return Indicator(
        values,
        periods,
    )















# ==========================================================
# BUILD ADX
# ==========================================================

adx = build_adx(
    high,
    low,
    close,
)

print("ADX Shape:", adx.values.shape)
print("ADX 14:", adx.get(14)[:20])











# ==========================================================
# ADX FEATURES
# ==========================================================

with Timer("ADX Features"):

    features.add(
        "adx",
        adx.values,
    )

print(f"Total Features: {features.count}")




















































# ==========================================================
# DISTANCE NORMALIZATION ENGINE
# ==========================================================

@njit(parallel=True, cache=True)
def normalized_distance(price, indicator, normalizer):

    rows = indicator.shape[0]
    cols = indicator.shape[1]

    out = np.empty((rows, cols), dtype=np.float32)

    for p in prange(cols):

        for i in range(rows):

            n = normalizer[i, p]

            if n == 0.0:

                out[i, p] = 0.0

            else:

                out[i, p] = (
                    price[i] - indicator[i, p]
                ) / n

    return out


































# ==========================================================
# EMA ATR DISTANCE
# ==========================================================

with Timer("EMA ATR Distance"):

    ema_atr = normalized_distance(
        close,
        ema.values,
        atr.values,
    )

features.add(
    "ema_atr",
    ema_atr,
)

print(f"Total Features: {features.count}")











# ==========================================================
# EMA STD DISTANCE
# ==========================================================

with Timer("EMA STD Distance"):

    ema_std = normalized_distance(
        close,
        ema.values,
        std.values,
    )

features.add(
    "ema_std",
    ema_std,
)

print(f"Total Features: {features.count}")































# ==========================================================
# DEMA ATR DISTANCE
# ==========================================================

with Timer("DEMA ATR Distance"):

    dema_atr = normalized_distance(
        close,
        dema.values,
        atr.values,
    )

features.add(
    "dema_atr",
    dema_atr,
)

print(f"Total Features: {features.count}")











# ==========================================================
# DEMA STD DISTANCE
# ==========================================================

with Timer("DEMA STD Distance"):

    dema_std = normalized_distance(
        close,
        dema.values,
        std.values,
    )

features.add(
    "dema_std",
    dema_std,
)

print(f"Total Features: {features.count}")


































# ==========================================================
# VWAP ATR DISTANCE
# ==========================================================

with Timer("VWAP ATR Distance"):

    vwap_atr = normalized_distance(
        close,
        vwap.values,
        atr.values[:, :1],
    )

features.add(
    "vwap_atr",
    vwap_atr,
)

print(f"Total Features: {features.count}")














# ==========================================================
# VWAP STD DISTANCE
# ==========================================================

with Timer("VWAP STD Distance"):

    vwap_std = normalized_distance(
        close,
        vwap.values,
        std.values[:, :1],
    )

features.add(
    "vwap_std",
    vwap_std,
)

print(f"Total Features: {features.count}")









































































































# ==========================================================
# CLOSE PRICE PIVOT ENGINE
# ==========================================================

@njit(cache=True)
def _pivot_events_single(close, period):

    rows = close.shape[0]

    highs = np.zeros(rows, dtype=np.int8)
    lows = np.zeros(rows, dtype=np.int8)

    for i in range(rows):

        value = close[i]

        is_high = True
        is_low = True

        # -------------------------
        # Left Side
        # -------------------------

        for j in range(1, period + 1):

            idx = i - j

            if idx >= 0:

                if close[idx] >= value:
                    is_high = False

                if close[idx] <= value:
                    is_low = False

        # -------------------------
        # Right Side
        # -------------------------

        for j in range(1, period + 1):

            idx = i + j

            if idx < rows:

                if close[idx] > value:
                    is_high = False

                if close[idx] < value:
                    is_low = False

        if is_high:
            highs[i] = 1

        if is_low:
            lows[i] = 1

    return highs, lows


def build_price_pivots(close):

    with Timer("Building Price Pivots"):

        high_events = {}
        low_events = {}

        for period in PERIODS:

            h, l = _pivot_events_single(
                close,
                period,
            )

            high_events[period] = h
            low_events[period] = l

    return high_events, low_events
















# ==========================================================
# BUILD PRICE PIVOTS
# ==========================================================

pivot_high_events, pivot_low_events = build_price_pivots(close)

print()

for p in PERIODS:

    print(
        p,
        pivot_high_events[p].sum(),
        pivot_low_events[p].sum(),
    )













































# ==========================================================
# PRICE PIVOT FEATURES
# ==========================================================

@njit(cache=True)
def _pivot_features(close, high_events, low_events):

    rows = close.shape[0]

    high_distance = np.zeros(rows, dtype=FLOAT)
    low_distance  = np.zeros(rows, dtype=FLOAT)

    bars_high = np.full(rows, -1, dtype=np.int32)
    bars_low  = np.full(rows, -1, dtype=np.int32)

    last_high_price = 0.0
    last_low_price = 0.0

    last_high_bar = -1
    last_low_bar = -1

    for i in range(rows):

        # -------------------------
        # Update pivot high
        # -------------------------

        if high_events[i]:

            last_high_price = close[i]
            last_high_bar = i

        if last_high_bar != -1:

            high_distance[i] = (
                (close[i] - last_high_price)
                / last_high_price
                * 100.0
            )

            bars_high[i] = i - last_high_bar

        # -------------------------
        # Update pivot low
        # -------------------------

        if low_events[i]:

            last_low_price = close[i]
            last_low_bar = i

        if last_low_bar != -1:

            low_distance[i] = (
                (close[i] - last_low_price)
                / last_low_price
                * 100.0
            )

            bars_low[i] = i - last_low_bar

    return (
        high_distance,
        low_distance,
        bars_high,
        bars_low,
    )


with Timer("Price Pivot Features"):

    for period in PERIODS:

        (
            high_dist,
            low_dist,
            bars_high,
            bars_low,
        ) = _pivot_features(
            close,
            pivot_high_events[period],
            pivot_low_events[period],
        )

        features.add(
            f"pivot_high_distance_{period}",
            high_dist,
        )

        features.add(
            f"pivot_low_distance_{period}",
            low_dist,
        )

        features.add(
            f"pivot_high_bars_{period}",
            bars_high,
        )

        features.add(
            f"pivot_low_bars_{period}",
            bars_low,
        )

print("Total Features:", features.count)

































# ==========================================================
# BUILD VOLUME PROFILE
# ==========================================================

VOLUME_PROFILE_LOOKBACK = 200
VOLUME_PROFILE_ROWS = 100


@njit(cache=True)
def build_volume_profile(
    high,
    low,
    close,
    volume,
):

    rows = close.shape[0]

    vp_rows = np.zeros(
        (rows, VOLUME_PROFILE_ROWS),
        dtype=np.float32,
    )

    vp_low = np.zeros(rows, dtype=np.float32)
    vp_high = np.zeros(rows, dtype=np.float32)
    vp_step = np.zeros(rows, dtype=np.float32)

    for i in range(rows):

        start = i - VOLUME_PROFILE_LOOKBACK + 1

        if start < 0:
            start = 0

        # -------------------------
        # Range
        # -------------------------

        lo = low[start]
        hi = high[start]

        for j in range(start + 1, i + 1):

            if low[j] < lo:
                lo = low[j]

            if high[j] > hi:
                hi = high[j]

        step = (hi - lo) / VOLUME_PROFILE_ROWS

        if step <= 0:

            vp_low[i] = lo
            vp_high[i] = hi
            vp_step[i] = 1.0

            continue

        vp_low[i] = lo
        vp_high[i] = hi
        vp_step[i] = step

        # -------------------------
        # Fill Histogram
        # -------------------------

        for j in range(start, i + 1):

            row = int((close[j] - lo) / step)

            if row < 0:
                row = 0

            elif row >= VOLUME_PROFILE_ROWS:
                row = VOLUME_PROFILE_ROWS - 1

            vp_rows[i, row] += volume[j]

    return (
        vp_rows,
        vp_low,
        vp_high,
        vp_step,
    )


with Timer("Building Volume Profile"):

    (
        vp_rows,
        vp_low,
        vp_high,
        vp_step,
    ) = build_volume_profile(
        high,
        low,
        close,
        volume,
    )

print("Volume Profile Shape:", vp_rows.shape)
print(vp_rows[:2, :10])


















# ==========================================================
# VOLUME PROFILE POC
# ==========================================================

@njit(cache=True)
def build_poc(
    vp_rows,
    vp_low,
    vp_step,
    close,
):

    rows = vp_rows.shape[0]

    poc_price = np.zeros(rows, dtype=np.float32)
    poc_distance = np.zeros(rows, dtype=np.float32)

    for i in range(rows):

        best_row = 0
        best_volume = vp_rows[i, 0]

        for r in range(1, VOLUME_PROFILE_ROWS):

            if vp_rows[i, r] > best_volume:

                best_volume = vp_rows[i, r]
                best_row = r

        price = vp_low[i] + (best_row + 0.5) * vp_step[i]

        poc_price[i] = price

        if price != 0.0:

            poc_distance[i] = (
                (close[i] - price)
                / price
                * 100.0
            )

    return (
        poc_price,
        poc_distance,
    )


with Timer("Building VP POC"):

    (
        vp_poc_price,
        vp_poc_distance,
    ) = build_poc(
        vp_rows,
        vp_low,
        vp_step,
        close,
    )

features.add(
    "vp_poc_distance",
    vp_poc_distance,
)

print("Total Features:", features.count)










































# ==========================================================
# VOLUME PROFILE ROW PIVOTS
# ==========================================================

VP_PIVOT_PERIODS = [1, 2, 4, 8]


@njit(cache=True)
def build_vp_row_pivots(vp_rows, period):

    bars = vp_rows.shape[0]
    rows = vp_rows.shape[1]

    pivots = np.zeros((bars, rows), dtype=np.uint8)

    for i in range(bars):

        for r in range(rows):

            value = vp_rows[i, r]

            good = True

            for j in range(1, period + 1):

                left = r - j
                right = r + j

                if left >= 0:
                    if value <= vp_rows[i, left]:
                        good = False
                        break

                # Missing rows beyond the edge are assumed zero.
                if right < rows:
                    if value <= vp_rows[i, right]:
                        good = False
                        break

            if good:
                pivots[i, r] = 1

    return pivots


vp_row_pivots = {}

for period in VP_PIVOT_PERIODS:

    with Timer(f"VP Row Pivots {period}"):

        vp_row_pivots[period] = build_vp_row_pivots(
            vp_rows,
            period,
        )

    print(
        period,
        vp_row_pivots[period].sum(),
    )





































# ==========================================================
# VOLUME PROFILE PIVOT FEATURES
# ==========================================================

@njit(cache=True)
def build_vp_pivot_features(
    vp_rows,
    pivots,
    vp_low,
    vp_step,
    close,
):

    bars = vp_rows.shape[0]
    rows = vp_rows.shape[1]

    current_value = np.zeros(bars, dtype=np.float32)

    upper_distance = np.zeros(bars, dtype=np.float32)
    lower_distance = np.zeros(bars, dtype=np.float32)

    upper_value = np.zeros(bars, dtype=np.float32)
    lower_value = np.zeros(bars, dtype=np.float32)

    for i in range(bars):

        # --------------------------------------------------
        # Current row
        # --------------------------------------------------

        row = int((close[i] - vp_low[i]) / vp_step[i])

        if row < 0:
            row = 0

        if row >= rows:
            row = rows - 1

        # normalize current row volume
        max_vol = 0.0

        for r in range(rows):

            if vp_rows[i, r] > max_vol:
                max_vol = vp_rows[i, r]

        if max_vol > 0:

            current_value[i] = (
                vp_rows[i, row]
                / max_vol
                * 100.0
            )

        # --------------------------------------------------
        # Search above
        # --------------------------------------------------

        upper = rows - 1

        for r in range(row + 1, rows):

            if pivots[i, r]:

                upper = r
                break

        # --------------------------------------------------
        # Search below
        # --------------------------------------------------

        lower = 0

        for r in range(row - 1, -1, -1):

            if pivots[i, r]:

                lower = r
                break

        upper_price = vp_low[i] + (upper + 0.5) * vp_step[i]
        lower_price = vp_low[i] + (lower + 0.5) * vp_step[i]

        if upper_price != 0:

            upper_distance[i] = (
                (upper_price - close[i])
                / upper_price
                * 100.0
            )

        if lower_price != 0:

            lower_distance[i] = (
                (close[i] - lower_price)
                / lower_price
                * 100.0
            )

        if max_vol > 0:

            upper_value[i] = (
                vp_rows[i, upper]
                / max_vol
                * 100.0
            )

            lower_value[i] = (
                vp_rows[i, lower]
                / max_vol
                * 100.0
            )

    return (
        current_value,
        upper_distance,
        lower_distance,
        upper_value,
        lower_value,
    )


print("Building VP Pivot Features")

for period in VP_PIVOT_PERIODS:

    with Timer(f"VP Pivot Features {period}"):

        (
            current_value,
            upper_distance,
            lower_distance,
            upper_value,
            lower_value,
        ) = build_vp_pivot_features(
            vp_rows,
            vp_row_pivots[period],
            vp_low,
            vp_step,
            close,
        )

    features.add(
        f"vp_current_row_value_{period}",
        current_value,
    )

    features.add(
        f"vp_upper_pivot_distance_{period}",
        upper_distance,
    )

    features.add(
        f"vp_lower_pivot_distance_{period}",
        lower_distance,
    )

    features.add(
        f"vp_upper_pivot_value_{period}",
        upper_value,
    )

    features.add(
        f"vp_lower_pivot_value_{period}",
        lower_value,
    )

    print("Total Features:", features.count)

































































































# ==========================================================
# SIGNAL CONFIG
# ==========================================================

FEATURE_FILE = "indicator_features.parquet"
SIGNAL_FILE = "signal_labels.parquet"

# ----------------------------------------------------------
# LIQUIDITY
# ----------------------------------------------------------

MIN_AVG_DOLLAR_VOLUME = 10_000.0
DOLLAR_VOLUME_LOOKBACK = 20

# ----------------------------------------------------------
# ATR SIGNALS
# ----------------------------------------------------------

ATR_TAKE_PROFITS = np.array(
    [
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
    ],
    dtype=np.float32,
)

ATR_STOP_LOSSES = np.array(
    [
        0.5,
        1.0,
    ],
    dtype=np.float32,
)

ATR_MAX_TAKE_PROFIT_PERCENT = np.float32(20.0)

# ----------------------------------------------------------
# PERCENT SIGNALS
# ----------------------------------------------------------

PERCENT_TAKE_PROFITS = np.array(
    [
        1.5,
        2.0,
        4.0,
        5.0,
        10.0,
        20.0,
    ],
    dtype=np.float32,
)

PERCENT_STOP_LOSSES = np.array(
    [
        0.5,
        1.5,
        2.0,
        5.0,
    ],
    dtype=np.float32,
)

# ----------------------------------------------------------
# COSTS
# ----------------------------------------------------------

TRADE_COST = np.float32(1.0)
EMERGENCY_STOP = np.float32(5.0)

# ==========================================================
# SIGNAL MATRIX
# ==========================================================

class SignalMatrix:

    def __init__(self):

        self.data = {}
        self.count = 0

    def add_matrix(
        self,
        prefix,
        matrix,
        tp_values,
        sl_values,
        direction,
    ):

        cols = 0

        for sl in sl_values:

            for tp in tp_values:

                self.data[
                    f"{prefix}_{direction}_tp{tp:g}_sl{sl:g}"
                ] = matrix[:, cols]

                cols += 1

                self.count += 1

    def dataframe(self):

        return pd.DataFrame(self.data)


signals = SignalMatrix()














# ==========================================================
# ATR SIGNAL ENGINE
# ==========================================================

ATR_COMBINATIONS = (
    len(ATR_TAKE_PROFITS)
    * len(ATR_STOP_LOSSES)
)


@njit(parallel=True, cache=True)
def build_atr_signal_matrix(
    close,
    high,
    low,
    atr,
    tp_values,
    sl_values,
    emergency_stop,
    trade_cost,
    max_tp_percent,
    direction,
):

    rows = close.shape[0]

    total = (
        tp_values.shape[0]
        * sl_values.shape[0]
    )

    signals = np.zeros(
        (rows, total),
        dtype=np.int8,
    )

    for entry_bar in prange(rows):

        entry = close[entry_bar]

        # --------------------------------------------
        # Build all targets once
        # --------------------------------------------

        tp_price = np.empty(total, dtype=np.float32)
        sl_price = np.empty(total, dtype=np.float32)

        finished = np.zeros(total, dtype=np.uint8)

        index = 0

        for sl in sl_values:

            for tp in tp_values:

                tp_distance = tp * atr[entry_bar]

                max_tp = (
                    entry
                    * max_tp_percent
                    / 100.0
                )

                if tp_distance > max_tp:
                    tp_distance = max_tp

                sl_distance = sl * atr[entry_bar]

                if direction == 1:

                    tp_price[index] = (
                        entry + tp_distance
                    )

                    sl_price[index] = (
                        entry - sl_distance
                    )

                else:

                    tp_price[index] = (
                        entry - tp_distance
                    )

                    sl_price[index] = (
                        entry + sl_distance
                    )

                index += 1

        # --------------------------------------------
        # Walk forward ONCE
        # --------------------------------------------

        active = total

        for future in range(
            entry_bar + 1,
            rows,
        ):

            h = high[future]
            l = low[future]

            if active == 0:
                break

            for k in range(total):

                if finished[k]:
                    continue

                if direction == 1:

                    emergency = (
                        entry
                        * (1.0 - emergency_stop / 100.0)
                    )

                    if l <= emergency:

                        finished[k] = 1
                        active -= 1
                        continue

                    if l <= sl_price[k]:

                        finished[k] = 1
                        active -= 1
                        continue

                    if h >= tp_price[k]:

                        profit = (
                            (tp_price[k] - entry)
                            / entry
                        ) * 100.0

                        profit -= trade_cost

                        if profit > 0:

                            signals[
                                entry_bar,
                                k,
                            ] = 1

                        finished[k] = 1
                        active -= 1

                else:

                    emergency = (
                        entry
                        * (1.0 + emergency_stop / 100.0)
                    )

                    if h >= emergency:

                        finished[k] = 1
                        active -= 1
                        continue

                    if h >= sl_price[k]:

                        finished[k] = 1
                        active -= 1
                        continue

                    if l <= tp_price[k]:

                        profit = (
                            (entry - tp_price[k])
                            / entry
                        ) * 100.0

                        profit -= trade_cost

                        if profit > 0:

                            signals[
                                entry_bar,
                                k,
                            ] = 1

                        finished[k] = 1
                        active -= 1

    return signals
















# ==========================================================
# PERCENT SIGNAL ENGINE
# ==========================================================

PERCENT_COMBINATIONS = (
    len(PERCENT_TAKE_PROFITS)
    * len(PERCENT_STOP_LOSSES)
)


@njit(parallel=True, cache=True)
def build_percent_signal_matrix(
    close,
    high,
    low,
    tp_values,
    sl_values,
    emergency_stop,
    trade_cost,
    direction,
):

    rows = close.shape[0]

    total = (
        tp_values.shape[0]
        * sl_values.shape[0]
    )

    signals = np.zeros(
        (rows, total),
        dtype=np.int8,
    )

    for entry_bar in prange(rows):

        entry = close[entry_bar]

        tp_price = np.empty(
            total,
            dtype=np.float32,
        )

        sl_price = np.empty(
            total,
            dtype=np.float32,
        )

        finished = np.zeros(
            total,
            dtype=np.uint8,
        )

        index = 0

        # ------------------------------------------
        # Build all TP / SL prices
        # ------------------------------------------

        for sl in sl_values:

            for tp in tp_values:

                if direction == 1:

                    tp_price[index] = (
                        entry
                        * (1.0 + tp / 100.0)
                    )

                    sl_price[index] = (
                        entry
                        * (1.0 - sl / 100.0)
                    )

                else:

                    tp_price[index] = (
                        entry
                        * (1.0 - tp / 100.0)
                    )

                    sl_price[index] = (
                        entry
                        * (1.0 + sl / 100.0)
                    )

                index += 1

        active = total

        # ------------------------------------------
        # Walk forward once
        # ------------------------------------------

        for future in range(
            entry_bar + 1,
            rows,
        ):

            if active == 0:
                break

            h = high[future]
            l = low[future]

            for k in range(total):

                if finished[k]:
                    continue

                # --------------------------------------
                # LONG
                # --------------------------------------

                if direction == 1:

                    emergency = (
                        entry
                        * (
                            1.0
                            - emergency_stop / 100.0
                        )
                    )

                    # Emergency stop

                    if l <= emergency:

                        finished[k] = 1
                        active -= 1
                        continue

                    # Stop loss

                    if l <= sl_price[k]:

                        finished[k] = 1
                        active -= 1
                        continue

                    # Take profit

                    if h >= tp_price[k]:

                        profit = (
                            (tp_price[k] - entry)
                            / entry
                        ) * 100.0

                        profit -= trade_cost

                        if profit > 0.0:

                            signals[
                                entry_bar,
                                k,
                            ] = 1

                        finished[k] = 1
                        active -= 1

                # --------------------------------------
                # SHORT
                # --------------------------------------

                else:

                    emergency = (
                        entry
                        * (
                            1.0
                            + emergency_stop / 100.0
                        )
                    )

                    # Emergency stop

                    if h >= emergency:

                        finished[k] = 1
                        active -= 1
                        continue
                    
                    # Stop loss

                    if h >= sl_price[k]:

                        finished[k] = 1
                        active -= 1
                        continue

                    # Take profit

                    if l <= tp_price[k]:

                        profit = (
                            (entry - tp_price[k])
                            / entry
                        ) * 100.0

                        profit -= trade_cost

                        if profit > 0.0:

                            signals[
                                entry_bar,
                                k,
                            ] = 1

                        finished[k] = 1
                        active -= 1

    return signals










# ==========================================================
# BUILD SIGNAL LABELS
# ==========================================================

print("Building Signal Labels...")

with Timer("Signal Labels"):

    # atr14 = atr.get(14)

    # # ------------------------------------------------------
    # # ATR LONG
    # # ------------------------------------------------------

    # atr_long = build_atr_signal_matrix(
    #     close,
    #     high,
    #     low,
    #     atr14,
    #     ATR_TAKE_PROFITS,
    #     ATR_STOP_LOSSES,
    #     EMERGENCY_STOP,
    #     TRADE_COST,
    #     ATR_MAX_TAKE_PROFIT_PERCENT,
    #     1,
    # )

    # signals.add_matrix(
    #     "signal_atr",
    #     atr_long,
    #     ATR_TAKE_PROFITS,
    #     ATR_STOP_LOSSES,
    #     "long",
    # )

    # # ------------------------------------------------------
    # # ATR SHORT
    # # ------------------------------------------------------

    # atr_short = build_atr_signal_matrix(
    #     close,
    #     high,
    #     low,
    #     atr14,
    #     ATR_TAKE_PROFITS,
    #     ATR_STOP_LOSSES,
    #     EMERGENCY_STOP,
    #     TRADE_COST,
    #     ATR_MAX_TAKE_PROFIT_PERCENT,
    #     -1,
    # )

    # signals.add_matrix(
    #     "signal_atr",
    #     atr_short,
    #     ATR_TAKE_PROFITS,
    #     ATR_STOP_LOSSES,
    #     "short",
    # )

    # ------------------------------------------------------
    # PERCENT LONG
    # ------------------------------------------------------

    percent_long = build_percent_signal_matrix(
        close,
        high,
        low,
        PERCENT_TAKE_PROFITS,
        PERCENT_STOP_LOSSES,
        EMERGENCY_STOP,
        TRADE_COST,
        1,
    )

    signals.add_matrix(
        "signal_pct",
        percent_long,
        PERCENT_TAKE_PROFITS,
        PERCENT_STOP_LOSSES,
        "long",
    )

    # ------------------------------------------------------
    # PERCENT SHORT
    # ------------------------------------------------------

    percent_short = build_percent_signal_matrix(
        close,
        high,
        low,
        PERCENT_TAKE_PROFITS,
        PERCENT_STOP_LOSSES,
        EMERGENCY_STOP,
        TRADE_COST,
        -1,
    )

    signals.add_matrix(
        "signal_pct",
        percent_short,
        PERCENT_TAKE_PROFITS,
        PERCENT_STOP_LOSSES,
        "short",
    )

print()

print(f"Signal Columns : {signals.count:,}")

















# ==========================================================
# SAVE FEATURES & SIGNALS
# ==========================================================

print()
print("=" * 60)
print("Building DataFrames")
print("=" * 60)

with Timer("Input DataFrame"):

    input_df = features.dataframe()

with Timer("Signal DataFrame"):

    signal_df = signals.dataframe()

    if "timestamp" in globals():

        signal_df.insert(
            0,
            "timestamp",
            timestamp,
        )

print()

print("Input Shape :", input_df.shape)
print("Signal Shape:", signal_df.shape)

print()

# ----------------------------------------------------------
# SAVE LOCALLY
# ----------------------------------------------------------

with Timer("Save Input"):

    input_df.to_parquet(
        INPUT_OUTPUT,
        compression="snappy",
        index=False,
    )

with Timer("Save Signals"):

    signal_df.to_parquet(
        SIGNAL_OUTPUT,
        compression="snappy",
        index=False,
    )

# ----------------------------------------------------------
# UPLOAD TO R2
# ----------------------------------------------------------

with Timer("Upload Results"):

    upload_results(SYMBOL)

print()

print("=" * 60)
print("Finished!")
print("=" * 60)

print(f"Stock   : {SYMBOL}")
print(f"Rows    : {len(input_df):,}")
print(f"Inputs  : {len(input_df.columns):,}")
print(f"Signals : {len(signal_df.columns):,}")

print()
print(f"Uploaded : input/{SYMBOL}.parquet")
print(f"Uploaded : output/{SYMBOL}.parquet")

print("=" * 60)



# ==========================================================
# CLEANUP
# ==========================================================

for file in (
    LOCAL_PARQUET,
    SYMBOLS_FILE,
    INPUT_OUTPUT,
    SIGNAL_OUTPUT,
):

    try:
        file.unlink()
    except FileNotFoundError:
        pass







































# # ==========================================================
# # SIGNAL CONFIG
# # ==========================================================

# # ----------------------------------------------------------
# # STOP LOSS MULTIPLIERS (ATR)
# # ----------------------------------------------------------

# STOP_LOSSES = np.arange(
#     0.5,
#     1.5,
#     0.5,
#     dtype=np.float32,
# )

# # ----------------------------------------------------------
# # TAKE PROFIT MULTIPLIERS (ATR)
# # ----------------------------------------------------------

# TAKE_PROFITS = np.arange(
#     1.0,
#     10.5,
#     0.5,
#     dtype=np.float32,
# )

# # ----------------------------------------------------------
# # STANDARD DEVIATION CROSS LEVELS
# # ----------------------------------------------------------

# STD_LEVELS = np.array(
#     [1.0, 2.0],
#     dtype=np.float32,
# )










# # ==========================================================
# # SIGNAL EVENT ENGINE
# # ==========================================================

# signal_events = {}

# # ----------------------------------------------------------
# # EMA CROSS EVENTS
# # ----------------------------------------------------------

# print("Building EMA Signal Events...")

# for period in ema.periods:

#     signal_events[f"close_cross_ema_{period}"] = cross_events(
#         close,
#         ema.get(period),
#     )

# # ----------------------------------------------------------
# # VOLUME PROFILE PIVOT CROSS EVENTS
# # ----------------------------------------------------------

# print("Building Volume Profile Signal Events...")

# for period in VP_PIVOT_PERIODS:

#     pivots = vp_row_pivots[period]

#     rows = pivots.shape[1]

#     events = np.zeros(ROWS, dtype=np.int8)

#     for r in range(rows):

#         prices = (
#             vp_low
#             + (r + 0.5) * vp_step
#         )

#         pivot_mask = pivots[:, r]

#         cross = cross_events(
#             close,
#             prices,
#         )

#         events = np.where(
#             (pivot_mask == 1) & (cross != 0),
#             cross,
#             events,
#         )

#     signal_events[f"vp_pivot_cross_{period}"] = events

# # ----------------------------------------------------------
# # EMA ±1 STD / ±2 STD CROSS EVENTS
# # ----------------------------------------------------------

# print("Building EMA STD Signal Events...")

# for period in ema.periods:

#     ema_values = ema.get(period)

#     std_values = std.get(period)

#     for level in STD_LEVELS:

#         upper = ema_values + std_values * level
#         lower = ema_values - std_values * level

#         signal_events[
#             f"ema_{period}_std_{level:g}_upper"
#         ] = cross_events(
#             close,
#             upper,
#         )

#         signal_events[
#             f"ema_{period}_std_{level:g}_lower"
#         ] = cross_events(
#             close,
#             lower,
#         )

# print()
# print(f"Total Signal Event Types: {len(signal_events)}")



























# # ==========================================================
# # SIGNAL LABEL ENGINE
# # ==========================================================

# @njit(cache=True)
# def build_signal_labels(
#     events,
#     close,
#     high,
#     low,
#     atr,
#     take_profit,
#     stop_loss,
# ):

#     rows = close.shape[0]

#     signal = np.full(rows, -1, dtype=np.int8)

#     for i in range(rows):

#         direction = events[i]

#         if direction == 0:
#             continue

#         entry = close[i]

#         risk = atr[i]

#         if risk <= 0.0:

#             signal[i] = 0
#             continue

#         # ------------------------------------------
#         # LONG
#         # ------------------------------------------

#         if direction == 1:

#             tp = entry + risk * take_profit
#             sl = entry - risk * stop_loss

#             result = 0

#             for j in range(i + 1, rows):

#                 if low[j] <= sl:

#                     result = 0
#                     break

#                 if high[j] >= tp:

#                     result = 1
#                     break

#             signal[i] = result

#         # ------------------------------------------
#         # SHORT
#         # ------------------------------------------

#         else:

#             tp = entry - risk * take_profit
#             sl = entry + risk * stop_loss

#             result = 0

#             for j in range(i + 1, rows):

#                 if high[j] >= sl:

#                     result = 0
#                     break

#                 if low[j] <= tp:

#                     result = 1
#                     break

#             signal[i] = result

#     return signal












# # ==========================================================
# # SIGNAL MATRIX
# # ==========================================================

# signals = FeatureMatrix()














# # ==========================================================
# # BUILD SIGNAL LABELS
# # ==========================================================

# print("Building Signal Labels...")

# with Timer("Signal Labels"):

#     total = 0

#     # ATR used for TP / SL sizing
#     atr_values = atr.get(14)

#     for event_name, event_array in signal_events.items():

#         for stop_loss in STOP_LOSSES:

#             for take_profit in TAKE_PROFITS:

#                 signal = build_signal_labels(
#                     event_array,
#                     close,
#                     high,
#                     low,
#                     atr_values,
#                     take_profit,
#                     stop_loss,
#                 )

#                 signals.add(
#                     (
#                         f"signal_{event_name}"
#                         f"_tp{take_profit:g}"
#                         f"_sl{stop_loss:g}"
#                     ),
#                     signal,
#                 )

#                 total += 1

# print()
# print(f"Signal Columns : {signals.count}")
# print(f"Signal Variants: {total}")













# # ==========================================================
# # SAVE FEATURE MATRIX
# # ==========================================================

# print("Building Feature DataFrame...")

# with Timer("Feature DataFrame"):

#     feature_df = features.dataframe()

# print("Feature Shape:", feature_df.shape)

# print("Saving Feature Parquet...")

# with Timer("Save Features"):

#     feature_df.to_parquet(
#         "indicator_features.parquet",
#         index=False,
#         compression="snappy",
#     )














# # ==========================================================
# # SAVE SIGNAL MATRIX
# # ==========================================================

# print("Building Signal DataFrame...")

# with Timer("Signal DataFrame"):

#     signal_df = signals.dataframe()

# print("Signal Shape:", signal_df.shape)

# print("Saving Signal Parquet...")

# with Timer("Save Signals"):

#     signal_df.to_parquet(
#         "signal_labels.parquet",
#         index=False,
#         compression="snappy",
#     )

































# # ==========================================================
# # SAVE FEATURES
# # ==========================================================

# print("Building DataFrame...")

# with Timer("DataFrame"):

#     feature_df = features.dataframe()

# print("Shape:", feature_df.shape)
# print("Columns:", len(feature_df.columns))

# print("Saving Parquet...")

# with Timer("Save Parquet"):

#     feature_df.to_parquet(
#         "indicator_features.parquet",
#         index=False,
#         compression="snappy",
#     )

# print()
# print("=" * 60)
# print("Finished!")
# print("Rows     :", len(feature_df))
# print("Features :", len(feature_df.columns))
# print("File     : indicator_features.parquet")
# print("=" * 60)