import io
import os
import tempfile

import boto3
import numpy as np
import pandas as pd

from xgboost import XGBClassifier


# ============================================================
# R2 CONFIGURATION
# ============================================================

R2_ACCESS_KEY_ID = "00e18b0c16ecb3395cd6f7c8e0eb3554"
R2_SECRET_ACCESS_KEY = "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"
R2_ENDPOINT_URL = "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"

R2_BUCKET_NAME = "stocks-data"


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SYMBOL = "AAPL"

TRADE_TYPE = "long"

STOP_LOSS_PERCENTAGE = 0.5

RISK_REWARD_RATIO = 1.0

INPUT_TIMEZONE = "America/New_York"


# ============================================================
# DATA SPLIT
# ============================================================

TRAIN_END_DATE = "2024-12-31"

VALIDATION_START_DATE = "2025-01-01"


# ============================================================
# FEATURE CONFIGURATION
# ============================================================
#
# These are the ONLY XGBoost features.
#
# XGBoost sees these values on EVERY aligned bar.
#
# The EMA crossover is NOT used to filter training data.
# ============================================================

EMA_FAST_COLUMN = "ema_distance_21"

EMA_SLOW_COLUMN = "ema_distance_50"

RR_RETURN_COLUMN = "trade_return_percent"


# ============================================================
# XGBOOST CONFIGURATION
# ============================================================

MAX_ESTIMATORS = 2000

MAX_DEPTH = 6

LEARNING_RATE = 0.05

SUBSAMPLE = 0.8

COLSAMPLE_BYTREE = 0.8

RANDOM_STATE = 42

PREDICTION_THRESHOLD = 0.50

XGBOOST_DEVICE = "cuda"


# ============================================================
# OUTPUT LOCATION
# ============================================================

OUTPUT_ROOT = f"ema_xgboost/{SYMBOL}"


ALIGNED_DATA_PATH = (
    f"{OUTPUT_ROOT}/data/aligned_dataset.parquet"
)

TRAIN_SIGNALS_PATH = (
    f"{OUTPUT_ROOT}/data/train_signals.parquet"
)

VALIDATION_SIGNALS_PATH = (
    f"{OUTPUT_ROOT}/data/validation_signals.parquet"
)

MODEL_PATH = (
    f"{OUTPUT_ROOT}/model/xgboost_model.json"
)

TRAIN_RAW_PATH = (
    f"{OUTPUT_ROOT}/equity_curves/train_raw.parquet"
)

TRAIN_XGBOOST_PATH = (
    f"{OUTPUT_ROOT}/equity_curves/train_xgboost.parquet"
)

VALIDATION_RAW_PATH = (
    f"{OUTPUT_ROOT}/equity_curves/validation_raw.parquet"
)

VALIDATION_XGBOOST_PATH = (
    f"{OUTPUT_ROOT}/equity_curves/validation_xgboost.parquet"
)


# ============================================================
# R2 CLIENT
# ============================================================

def create_r2_client():

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )


# ============================================================
# R2 DOWNLOAD PARQUET
# ============================================================

def download_parquet(
    client,
    path,
):

    response = client.get_object(
        Bucket=R2_BUCKET_NAME,
        Key=path,
    )

    data = response["Body"].read()

    return pd.read_parquet(
        io.BytesIO(data)
    )


# ============================================================
# R2 UPLOAD PARQUET
# ============================================================

def upload_parquet(
    client,
    dataframe,
    path,
):

    buffer = io.BytesIO()

    dataframe.to_parquet(
        buffer,
        index=False,
    )

    buffer.seek(0)

    client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=path,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )


# ============================================================
# SOURCE PATHS
# ============================================================

def get_input_path():

    return (
        f"input/{SYMBOL}.parquet"
    )


def get_risk_reward_path():

    return (
        f"riskreward/"
        f"{SYMBOL}/"
        f"{TRADE_TYPE}/"
        f"sl_{float(STOP_LOSS_PERCENTAGE)}_"
        f"rr_{float(RISK_REWARD_RATIO)}.parquet"
    )


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input_data(
    input_data,
):

    data = input_data.copy()

    if "timestamp" not in data.columns:

        raise ValueError(
            "Input data does not contain "
            "'timestamp'."
        )

    data["timestamp"] = (
        pd.to_datetime(
            data["timestamp"],
            utc=True,
        )
        .dt.tz_convert(
            INPUT_TIMEZONE
        )
    )

    data = (
        data
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return data


# ============================================================
# PREPARE RISK/REWARD DATA
# ============================================================

def prepare_risk_reward_data(
    risk_reward_data,
):

    data = risk_reward_data.copy()

    if "start_timestamp" not in data.columns:

        raise ValueError(
            "Risk/reward data does not contain "
            "'start_timestamp'."
        )

    if RR_RETURN_COLUMN not in data.columns:

        raise ValueError(
            f"Risk/reward data does not contain "
            f"'{RR_RETURN_COLUMN}'."
        )

    data["start_timestamp"] = (
        pd.to_datetime(
            data["start_timestamp"],
            utc=True,
        )
        .dt.tz_convert(
            INPUT_TIMEZONE
        )
    )

    data = (
        data
        .sort_values(
            "start_timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return data


# ============================================================
# MATCH RR TO INPUT
# ============================================================
#
# For every RR bar:
#
#     RR timestamp - 5 minutes < input timestamp
#     input timestamp <= RR timestamp
#
# The latest valid input bar is selected.
#
# This allows the RR timestamp itself to match.
# ============================================================

def match_rr_to_input_rows(
    input_data,
    risk_reward_data,
):

    input_timestamps = pd.DatetimeIndex(
        input_data["timestamp"]
    )

    rr_timestamps = pd.DatetimeIndex(
        risk_reward_data["start_timestamp"]
    )

    input_ns = input_timestamps.asi8

    rr_ns = rr_timestamps.asi8

    five_minutes = (
        pd.Timedelta(
            minutes=5
        ).value
    )

    matched_indices = np.full(
        len(rr_timestamps),
        -1,
        dtype=np.int64,
    )

    for i in range(
        len(rr_timestamps)
    ):

        rr_end = rr_ns[i]

        rr_start = (
            rr_end
            -
            five_minutes
        )

        right = np.searchsorted(
            input_ns,
            rr_end,
            side="right",
        )

        left = np.searchsorted(
            input_ns,
            rr_start,
            side="right",
        )

        if right > left:

            matched_indices[i] = (
                right - 1
            )

    return matched_indices


# ============================================================
# BUILD ALIGNED DATASET
# ============================================================
#
# IMPORTANT:
#
# Every successfully matched RR bar remains in the dataset.
#
# XGBoost training therefore uses ALL aligned bars.
#
# EMA crossover is calculated separately and retained as a
# trading signal.
# ============================================================

def build_aligned_dataset(
    input_data,
    risk_reward_data,
):

    input_data = prepare_input_data(
        input_data
    )

    risk_reward_data = (
        prepare_risk_reward_data(
            risk_reward_data
        )
    )

    # --------------------------------------------------------
    # VERIFY EMA COLUMNS
    # --------------------------------------------------------

    if EMA_FAST_COLUMN not in input_data.columns:

        raise ValueError(
            f"{EMA_FAST_COLUMN} is missing "
            "from the input data."
        )

    if EMA_SLOW_COLUMN not in input_data.columns:

        raise ValueError(
            f"{EMA_SLOW_COLUMN} is missing "
            "from the input data."
        )

    # --------------------------------------------------------
    # XGBOOST FEATURES
    # --------------------------------------------------------

    feature_names = [
        EMA_FAST_COLUMN,
        EMA_SLOW_COLUMN,
    ]

    # --------------------------------------------------------
    # CALCULATE EMA CROSSOVER
    # --------------------------------------------------------
    #
    # This is calculated on the COMPLETE 1-minute
    # input dataset BEFORE RR alignment.
    #
    # Therefore shift(1) means the previous 1-minute
    # bar.
    #
    # Upward crossover:
    #
    # previous fast <= previous slow
    # current fast  >  current slow
    # --------------------------------------------------------

    fast = pd.to_numeric(
        input_data[
            EMA_FAST_COLUMN
        ],
        errors="coerce",
    )

    slow = pd.to_numeric(
        input_data[
            EMA_SLOW_COLUMN
        ],
        errors="coerce",
    )

    input_data["ema_crossover"] = (
        (fast.shift(1) <= slow.shift(1))
        &
        (fast > slow)
    )

    # --------------------------------------------------------
    # MATCH RR BARS
    # --------------------------------------------------------

    matched_indices = (
        match_rr_to_input_rows(
            input_data,
            risk_reward_data,
        )
    )

    valid_matches = (
        matched_indices >= 0
    )

    if not np.any(valid_matches):

        raise ValueError(
            "No risk/reward rows could be "
            "matched to input data."
        )

    # --------------------------------------------------------
    # KEEP ALL SUCCESSFULLY MATCHED RR BARS
    # --------------------------------------------------------

    rr = (
        risk_reward_data
        .loc[valid_matches]
        .reset_index(drop=True)
    )

    input_indices = (
        matched_indices[
            valid_matches
        ]
    )

    input_rows = (
        input_data
        .iloc[input_indices]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # BUILD DATASET
    # --------------------------------------------------------

    dataset = pd.DataFrame()

    dataset["timestamp"] = (
        rr[
            "start_timestamp"
        ].to_numpy()
    )

    # --------------------------------------------------------
    # XGBOOST FEATURES
    #
    # EVERY BAR receives both features.
    # --------------------------------------------------------

    for feature_name in feature_names:

        dataset[feature_name] = (
            input_rows[
                feature_name
            ].to_numpy()
        )

    # --------------------------------------------------------
    # ACTUAL RR RETURN
    # --------------------------------------------------------

    dataset[
        RR_RETURN_COLUMN
    ] = pd.to_numeric(
        rr[
            RR_RETURN_COLUMN
        ],
        errors="coerce",
    ).to_numpy()

    # --------------------------------------------------------
    # EMA SIGNAL
    #
    # Retained for trading.
    #
    # Does NOT remove rows from the dataset.
    # --------------------------------------------------------

    dataset[
        "ema_crossover"
    ] = (
        input_rows[
            "ema_crossover"
        ].to_numpy()
    )

    # --------------------------------------------------------
    # XGBOOST TARGET
    #
    # EVERY BAR receives a target.
    #
    # 1 = profitable RR trade
    # 0 = non-profitable RR trade
    # --------------------------------------------------------

    dataset["target"] = (
        dataset[
            RR_RETURN_COLUMN
        ] > 0.0
    ).astype(
        np.int8
    )

    # --------------------------------------------------------
    # REMOVE INVALID NUMERIC ROWS
    # --------------------------------------------------------

    numeric_columns = (
        feature_names
        +
        [RR_RETURN_COLUMN]
    )

    numeric_values = (
        dataset[
            numeric_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    valid_numeric = np.all(
        np.isfinite(
            numeric_values
        ),
        axis=1,
    )

    dataset = (
        dataset
        .loc[valid_numeric]
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    crossover_count = int(
        dataset[
            "ema_crossover"
        ].sum()
    )

    non_crossover_count = (
        len(dataset)
        -
        crossover_count
    )

    print()

    print(
        "XGBoost feature selection:"
    )

    print(
        f"  Selected features: "
        f"{len(feature_names)}"
    )

    for feature_name in feature_names:

        print(
            f"    {feature_name}"
        )

    print()

    print(
        "EMA trading signal:"
    )

    print(
        f"  Fast EMA: "
        f"{EMA_FAST_COLUMN}"
    )

    print(
        f"  Slow EMA: "
        f"{EMA_SLOW_COLUMN}"
    )

    print()

    print(
        f"Aligned rows: "
        f"{len(dataset):,}"
    )

    print(
        f"EMA crossover rows: "
        f"{crossover_count:,}"
    )

    print(
        f"Non-crossover rows: "
        f"{non_crossover_count:,}"
    )

    print()

    print(
        "XGBoost training:"
    )

    print(
        "  Uses ALL aligned bars."
    )

    print()

    print(
        "EMA + XGBoost trading:"
    )

    print(
        "  EMA crossover AND "
        "XGBoost probability >= "
        f"{PREDICTION_THRESHOLD:.2f}"
    )

    return (
        dataset,
        feature_names,
    )


# ============================================================
# SPLIT DATA
# ============================================================

def split_train_validation(
    dataset,
):

    train_end = pd.Timestamp(
        TRAIN_END_DATE,
        tz=INPUT_TIMEZONE,
    )

    validation_start = pd.Timestamp(
        VALIDATION_START_DATE,
        tz=INPUT_TIMEZONE,
    )

    train_mask = (
        dataset[
            "timestamp"
        ]
        <= train_end
    )

    validation_mask = (
        dataset[
            "timestamp"
        ]
        >= validation_start
    )

    train_data = (
        dataset
        .loc[train_mask]
        .reset_index(
            drop=True
        )
    )

    validation_data = (
        dataset
        .loc[validation_mask]
        .reset_index(
            drop=True
        )
    )

    return (
        train_data,
        validation_data,
    )


# ============================================================
# GET EMA SIGNAL MASK
# ============================================================
#
# This is ONLY used when deciding which bars become
# actual raw EMA trades.
#
# It is NOT used for XGBoost training.
# ============================================================

def get_ema_signal_mask(
    dataset,
):

    return (
        dataset[
            "ema_crossover"
        ]
        .fillna(False)
        .to_numpy(
            dtype=bool
        )
    )


# ============================================================
# BUILD TRAINING ARRAYS
# ============================================================
#
# ALL training rows are passed to XGBoost.
# ============================================================

def build_training_arrays(
    train_data,
    feature_names,
):

    X = (
        train_data[
            feature_names
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    y = (
        train_data[
            "target"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    valid = (
        np.all(
            np.isfinite(X),
            axis=1,
        )
        &
        np.isfinite(y)
    )

    X = X[valid]

    y = y[valid]

    if len(X) == 0:

        raise ValueError(
            "No valid training samples."
        )

    if len(np.unique(y)) < 2:

        raise ValueError(
            "Training data contains only "
            "one target class."
        )

    return (
        X,
        y,
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    return XGBClassifier(
        n_estimators=MAX_ESTIMATORS,

        max_depth=MAX_DEPTH,

        learning_rate=LEARNING_RATE,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE_BYTREE,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=RANDOM_STATE,

        tree_method="hist",

        device=XGBOOST_DEVICE,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    train_data,
    feature_names,
):

    X, y = build_training_arrays(
        train_data,
        feature_names,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "XGBOOST TRAINING"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Training samples: "
        f"{len(X):,}"
    )

    print(
        f"Features: "
        f"{X.shape[1]}"
    )

    for feature_name in feature_names:

        print(
            f"  {feature_name}"
        )

    print()

    print(
        f"Trees: "
        f"{MAX_ESTIMATORS}"
    )

    print(
        f"Max depth: "
        f"{MAX_DEPTH}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Device: "
        f"{XGBOOST_DEVICE}"
    )

    print()

    print(
        "Training on ALL bars."
    )

    model = create_model()

    model.fit(
        X,
        y,
        verbose=False,
    )

    return model


# ============================================================
# PREDICT PROBABILITIES
# ============================================================
#
# Predictions are generated for EVERY bar.
#
# We do NOT filter by EMA crossover here.
# ============================================================

def predict_probabilities(
    model,
    data,
    feature_names,
):

    if len(data) == 0:

        return np.empty(
            0,
            dtype=np.float32,
        )

    X = (
        data[
            feature_names
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    valid = np.all(
        np.isfinite(X),
        axis=1,
    )

    probabilities = np.full(
        len(data),
        np.nan,
        dtype=np.float32,
    )

    if np.any(valid):

        probabilities[valid] = (
            model
            .predict_proba(
                X[valid]
            )[:, 1]
        )

    return probabilities


# ============================================================
# BUILD EQUITY CURVE
# ============================================================

def build_equity_curve(
    data,
    trade_mask=None,
):

    if len(data) == 0:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "trade_return_percent",
                "equity",
            ]
        )

    data = (
        data
        .copy()
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # SELECT TRADES
    # --------------------------------------------------------

    if trade_mask is None:

        selected = data

    else:

        trade_mask = np.asarray(
            trade_mask,
            dtype=bool,
        )

        if len(trade_mask) != len(data):

            raise ValueError(
                "Trade mask length does not "
                "match data."
            )

        selected = (
            data
            .loc[trade_mask]
            .copy()
            .reset_index(
                drop=True
            )
        )

    if len(selected) == 0:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "trade_return_percent",
                "equity",
            ]
        )

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    returns = (
        pd.to_numeric(
            selected[
                RR_RETURN_COLUMN
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    valid = np.isfinite(
        returns
    )

    selected = (
        selected
        .loc[valid]
        .reset_index(
            drop=True
        )
    )

    returns = returns[valid]

    # --------------------------------------------------------
    # COMPOUND EQUITY
    # --------------------------------------------------------

    equity = np.cumprod(
        1.0 + returns / 100.0,
        dtype=np.float64,
    )

    return pd.DataFrame(
        {
            "timestamp": (
                selected[
                    "timestamp"
                ].to_numpy()
            ),

            "trade_return_percent": (
                returns
            ),

            "equity": (
                equity
            ),
        }
    )


# ============================================================
# CURVE SUMMARY
# ============================================================

def summarize_curve(
    equity_curve,
):

    if len(equity_curve) == 0:

        return {
            "start_equity": 1.0,
            "end_equity": 1.0,
            "trades": 0,
            "total_return_percent": 0.0,
            "win_rate_percent": 0.0,
        }

    returns = (
        equity_curve[
            "trade_return_percent"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    end_equity = float(
        equity_curve[
            "equity"
        ]
        .iloc[-1]
    )

    trades = len(
        returns
    )

    wins = np.sum(
        returns > 0.0
    )

    win_rate = (
        wins
        /
        trades
        *
        100.0
    )

    total_return = (
        end_equity
        -
        1.0
    ) * 100.0

    return {
        "start_equity": 1.0,
        "end_equity": end_equity,
        "trades": trades,
        "total_return_percent": total_return,
        "win_rate_percent": win_rate,
    }


# ============================================================
# PRINT CURVE
# ============================================================

def print_curve(
    name,
    equity_curve,
):

    summary = summarize_curve(
        equity_curve
    )

    print(
        f"\n{name}"
    )

    print(
        f"  Start Equity: "
        f"{summary['start_equity']:.6f}"
    )

    print(
        f"  End Equity:   "
        f"{summary['end_equity']:.6f}"
    )

    print(
        f"  Trades:       "
        f"{summary['trades']:,}"
    )

    print(
        f"  Total Return: "
        f"{summary['total_return_percent']:.2f}%"
    )

    print(
        f"  Win Rate:     "
        f"{summary['win_rate_percent']:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "EMA XGBOOST COMPLETE PIPELINE"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # R2 CLIENT
    # ========================================================

    client = create_r2_client()

    # ========================================================
    # DOWNLOAD INPUT DATA
    # ========================================================

    input_path = get_input_path()

    print(
        "\nDownloading input data:"
    )

    print(
        f"  {input_path}"
    )

    input_data = download_parquet(
        client,
        input_path,
    )

    print(
        f"  Rows: "
        f"{len(input_data):,}"
    )

    # ========================================================
    # DOWNLOAD RR DATA
    # ========================================================

    rr_path = get_risk_reward_path()

    print(
        "\nDownloading risk/reward data:"
    )

    print(
        f"  {rr_path}"
    )

    risk_reward_data = download_parquet(
        client,
        rr_path,
    )

    print(
        f"  Rows: "
        f"{len(risk_reward_data):,}"
    )

    # ========================================================
    # ALIGN DATA
    # ========================================================

    print(
        "\nBuilding aligned dataset..."
    )

    (
        dataset,
        feature_names,
    ) = build_aligned_dataset(
        input_data,
        risk_reward_data,
    )

    print(
        f"\nAligned rows: "
        f"{len(dataset):,}"
    )

    # ========================================================
    # SAVE ALIGNED DATA
    # ========================================================

    upload_parquet(
        client,
        dataset,
        ALIGNED_DATA_PATH,
    )

    print(
        "\nSaved aligned data:"
    )

    print(
        f"  {ALIGNED_DATA_PATH}"
    )

    # ========================================================
    # SPLIT
    # ========================================================

    (
        train_data,
        validation_data,
    ) = split_train_validation(
        dataset
    )

    # ========================================================
    # EMA SIGNAL COUNTS
    # ========================================================

    train_ema_mask = (
        get_ema_signal_mask(
            train_data
        )
    )

    validation_ema_mask = (
        get_ema_signal_mask(
            validation_data
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DATA SPLIT"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Training RR bars: "
        f"{len(train_data):,}"
    )

    print(
        f"Training EMA signals: "
        f"{np.sum(train_ema_mask):,}"
    )

    print()

    print(
        f"Validation RR bars: "
        f"{len(validation_data):,}"
    )

    print(
        f"Validation EMA signals: "
        f"{np.sum(validation_ema_mask):,}"
    )

    print()

    print(
        "XGBoost training uses:"
    )

    print(
        f"  ALL {len(train_data):,} training bars"
    )

    # ========================================================
    # SAVE TRAINING / VALIDATION DATA
    # ========================================================
    #
    # These files now contain ALL bars.
    #
    # The filenames are retained for compatibility.
    # ========================================================

    upload_parquet(
        client,
        train_data,
        TRAIN_SIGNALS_PATH,
    )

    upload_parquet(
        client,
        validation_data,
        VALIDATION_SIGNALS_PATH,
    )

    print(
        "\nSaved training/validation datasets."
    )

    # ========================================================
    # TRAIN XGBOOST
    # ========================================================

    model = train_model(
        train_data,
        feature_names,
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    with tempfile.NamedTemporaryFile(
        suffix=".json",
        delete=False,
    ) as temp_file:

        model_path = temp_file.name

    try:

        model.save_model(
            model_path
        )

        with open(
            model_path,
            "rb",
        ) as file:

            model_bytes = file.read()

        client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=MODEL_PATH,
            Body=model_bytes,
            ContentType="application/json",
        )

    finally:

        if os.path.exists(
            model_path
        ):

            os.remove(
                model_path
            )

    print(
        "\nSaved model:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    # ========================================================
    # XGBOOST PREDICTIONS
    #
    # Predictions are made on EVERY bar.
    # ========================================================

    train_probabilities = (
        predict_probabilities(
            model,
            train_data,
            feature_names,
        )
    )

    validation_probabilities = (
        predict_probabilities(
            model,
            validation_data,
            feature_names,
        )
    )

    # ========================================================
    # XGBOOST CONDITIONS
    # ========================================================
    #
    # XGBoost says YES when probability >= 0.50.
    #
    # This condition alone does NOT create a trade.
    # ========================================================

    train_xgboost_condition = (
        np.isfinite(
            train_probabilities
        )
        &
        (
            train_probabilities
            >= PREDICTION_THRESHOLD
        )
    )

    validation_xgboost_condition = (
        np.isfinite(
            validation_probabilities
        )
        &
        (
            validation_probabilities
            >= PREDICTION_THRESHOLD
        )
    )

    # ========================================================
    # FINAL TRADE CONDITIONS
    # ========================================================
    #
    # TRADE =
    #
    #     EMA crossover
    #     AND
    #     XGBoost says profitable
    #
    # This is the key global-filter architecture.
    # ========================================================

    train_trade_mask = (
        train_ema_mask
        &
        train_xgboost_condition
    )

    validation_trade_mask = (
        validation_ema_mask
        &
        validation_xgboost_condition
    )

    # ========================================================
    # BUILD RAW EMA CURVES
    # ========================================================
    #
    # Raw EMA trades EVERY EMA crossover.
    # ========================================================

    train_raw = build_equity_curve(
        train_data,
        train_ema_mask,
    )

    validation_raw = build_equity_curve(
        validation_data,
        validation_ema_mask,
    )

    # ========================================================
    # BUILD EMA + XGBOOST CURVES
    # ========================================================
    #
    # Trades only when:
    #
    #     EMA crossover
    #     AND
    #     XGBoost >= threshold
    # ========================================================

    train_xgboost = build_equity_curve(
        train_data,
        train_trade_mask,
    )

    validation_xgboost = build_equity_curve(
        validation_data,
        validation_trade_mask,
    )

    # ========================================================
    # PRINT EQUITY RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EQUITY CURVES"
    )

    print(
        "=" * 70
    )

    print_curve(
        "Train Raw EMA",
        train_raw,
    )

    print_curve(
        "Train EMA + XGBoost",
        train_xgboost,
    )

    print_curve(
        "Validation Raw EMA",
        validation_raw,
    )

    print_curve(
        "Validation EMA + XGBoost",
        validation_xgboost,
    )

    # ========================================================
    # FILTER SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "XGBOOST GLOBAL FILTER"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Prediction threshold: "
        f"{PREDICTION_THRESHOLD:.2f}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\nTraining:"
    )

    print(
        f"  Total bars: "
        f"{len(train_data):,}"
    )

    print(
        f"  EMA signals: "
        f"{np.sum(train_ema_mask):,}"
    )

    print(
        f"  XGBoost YES: "
        f"{np.sum(train_xgboost_condition):,}"
    )

    print(
        f"  Final trades: "
        f"{np.sum(train_trade_mask):,}"
    )

    print(
        f"  EMA signals rejected by XGBoost: "
        f"{np.sum(train_ema_mask & ~train_xgboost_condition):,}"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "\nValidation:"
    )

    print(
        f"  Total bars: "
        f"{len(validation_data):,}"
    )

    print(
        f"  EMA signals: "
        f"{np.sum(validation_ema_mask):,}"
    )

    print(
        f"  XGBoost YES: "
        f"{np.sum(validation_xgboost_condition):,}"
    )

    print(
        f"  Final trades: "
        f"{np.sum(validation_trade_mask):,}"
    )

    print(
        f"  EMA signals rejected by XGBoost: "
        f"{np.sum(validation_ema_mask & ~validation_xgboost_condition):,}"
    )

    # ========================================================
    # SAVE FOUR EQUITY CURVES
    # ========================================================

    curves = [
        (
            train_raw,
            TRAIN_RAW_PATH,
        ),
        (
            train_xgboost,
            TRAIN_XGBOOST_PATH,
        ),
        (
            validation_raw,
            VALIDATION_RAW_PATH,
        ),
        (
            validation_xgboost,
            VALIDATION_XGBOOST_PATH,
        ),
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "SAVING EQUITY CURVES"
    )

    print(
        "=" * 70
    )

    for curve, path in curves:

        upload_parquet(
            client,
            curve,
            path,
        )

        print(
            f"Saved: {path}"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PIPELINE COMPLETE"
    )

    print(
        "=" * 70
    )

    return {
        "dataset": dataset,
        "train_data": train_data,
        "validation_data": validation_data,
        "model": model,
        "train_raw": train_raw,
        "train_xgboost": train_xgboost,
        "validation_raw": validation_raw,
        "validation_xgboost": validation_xgboost,
    }


if __name__ == "__main__":

    main()
