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

FEATURE_START_COLUMN = 1

FEATURE_END_COLUMN = 28

EMA_FAST_COLUMN = "ema_distance_9"

EMA_SLOW_COLUMN = "ema_distance_21"

RR_RETURN_COLUMN = "trade_return_percent"


# ============================================================
# XGBOOST CONFIGURATION
# ============================================================

MAX_ESTIMATORS = 20

MAX_DEPTH = 3

LEARNING_RATE = 0.05

SUBSAMPLE = 0.8

COLSAMPLE_BYTREE = 0.8

RANDOM_STATE = 42

PREDICTION_THRESHOLD = 0.50

XGBOOST_DEVICE = "cuda"


# ============================================================
# NEW OUTPUT LOCATION
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
# DOWNLOAD MODEL
# ============================================================

def download_model(
    client,
):

    response = client.get_object(
        Bucket=R2_BUCKET_NAME,
        Key=MODEL_PATH,
    )

    model_bytes = response["Body"].read()

    model = XGBClassifier()

    with tempfile.NamedTemporaryFile(
        suffix=".json",
        delete=False,
    ) as temp_file:

        temp_path = temp_file.name

        temp_file.write(
            model_bytes
        )

    try:

        model.load_model(
            temp_path
        )

    finally:

        if os.path.exists(
            temp_path
        ):

            os.remove(
                temp_path
            )

    return model


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
# PREPARE RR DATA
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

        # Latest input timestamp <= RR timestamp.
        right = np.searchsorted(
            input_ns,
            rr_end,
            side="right",
        )

        # First input timestamp > RR timestamp - 5 minutes.
        #
        # Therefore the left edge is EXCLUSIVE.
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

    if len(input_data.columns) < FEATURE_END_COLUMN:

        raise ValueError(
            "Input data does not have enough "
            "columns for the 27 features."
        )

    feature_names = list(
        input_data.columns[
            FEATURE_START_COLUMN:
            FEATURE_END_COLUMN
        ]
    )

    if len(feature_names) != 27:

        raise ValueError(
            f"Expected 27 features, "
            f"got {len(feature_names)}."
        )

    if EMA_FAST_COLUMN not in feature_names:

        raise ValueError(
            f"{EMA_FAST_COLUMN} is not "
            f"in the first 27 features."
        )

    if EMA_SLOW_COLUMN not in feature_names:

        raise ValueError(
            f"{EMA_SLOW_COLUMN} is not "
            f"in the first 27 features."
        )

    matched_indices = (
        match_rr_to_input_rows(
            input_data,
            risk_reward_data,
        )
    )

    valid_matches = (
        matched_indices >= 0
    )

    rr = (
        risk_reward_data
        .loc[valid_matches]
        .reset_index(
            drop=True
        )
    )

    input_indices = (
        matched_indices[
            valid_matches
        ]
    )

    input_rows = (
        input_data
        .iloc[input_indices]
        .reset_index(
            drop=True
        )
    )

    dataset = pd.DataFrame()

    dataset["timestamp"] = (
        rr["start_timestamp"]
        .to_numpy()
    )

    for feature_name in feature_names:

        dataset[feature_name] = (
            input_rows[
                feature_name
            ]
            .to_numpy()
        )

    dataset[RR_RETURN_COLUMN] = (
        pd.to_numeric(
            rr[RR_RETURN_COLUMN],
            errors="coerce",
        )
        .to_numpy()
    )

    # --------------------------------------------------------
    # EMA CROSSOVER
    # --------------------------------------------------------

    fast = pd.to_numeric(
        dataset[
            EMA_FAST_COLUMN
        ],
        errors="coerce",
    )

    slow = pd.to_numeric(
        dataset[
            EMA_SLOW_COLUMN
        ],
        errors="coerce",
    )

    dataset["ema_crossover"] = (
        (fast.shift(1) <= slow.shift(1))
        &
        (fast > slow)
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    dataset["target"] = (
        dataset[
            RR_RETURN_COLUMN
        ]
        > 0.0
    ).astype(
        np.int8
    )

    # --------------------------------------------------------
    # REMOVE INVALID VALUES
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
        dataset["timestamp"]
        <= train_end
    )

    validation_mask = (
        dataset["timestamp"]
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
# GET EMA SIGNALS
# ============================================================

def get_ema_signal_data(
    dataset,
):

    return (
        dataset
        .loc[
            dataset[
                "ema_crossover"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# BUILD TRAINING ARRAYS
# ============================================================

def build_training_arrays(
    train_signal_data,
    feature_names,
):

    X = (
        train_signal_data[
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
        train_signal_data[
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

    return X, y


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
    train_signal_data,
    feature_names,
):

    X, y = build_training_arrays(
        train_signal_data,
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

    print(
        f"\nTraining samples: "
        f"{len(X):,}"
    )

    print(
        f"Features: "
        f"{X.shape[1]}"
    )

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

    model = create_model()

    model.fit(
        X,
        y,
        verbose=False,
    )

    return model


# ============================================================
# PREDICT
# ============================================================

def predict_probabilities(
    model,
    signal_data,
    feature_names,
):

    if len(signal_data) == 0:

        return np.empty(
            0,
            dtype=np.float32,
        )

    X = (
        signal_data[
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
        len(signal_data),
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
    signal_data,
    trade_mask=None,
):

    if len(signal_data) == 0:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "trade_return_percent",
                "equity",
            ]
        )

    data = (
        signal_data
        .copy()
        .reset_index(
            drop=True
        )
    )

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
                "match signal data."
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

    equity = np.cumprod(
        1.0 + returns / 100.0,
        dtype=np.float64,
    )

    return pd.DataFrame(
        {
            "timestamp": (
                selected[
                    "timestamp"
                ]
                .to_numpy()
            ),

            "trade_return_percent": returns,

            "equity": equity,
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
        end_equity - 1.0
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

    client = create_r2_client()

    # ========================================================
    # DOWNLOAD SOURCE DATA
    # ========================================================

    input_path = get_input_path()

    rr_path = get_risk_reward_path()

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
        f"Aligned rows: "
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
        f"Saved aligned data:"
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
    # SIGNALS
    # ========================================================

    train_signals = (
        get_ema_signal_data(
            train_data
        )
    )

    validation_signals = (
        get_ema_signal_data(
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

    print(
        f"\nTraining RR bars: "
        f"{len(train_data):,}"
    )

    print(
        f"Training EMA signals: "
        f"{len(train_signals):,}"
    )

    print(
        f"\nValidation RR bars: "
        f"{len(validation_data):,}"
    )

    print(
        f"Validation EMA signals: "
        f"{len(validation_signals):,}"
    )

    # ========================================================
    # SAVE SIGNAL DATA
    # ========================================================

    upload_parquet(
        client,
        train_signals,
        TRAIN_SIGNALS_PATH,
    )

    upload_parquet(
        client,
        validation_signals,
        VALIDATION_SIGNALS_PATH,
    )

    print(
        "\nSaved signal datasets."
    )

    # ========================================================
    # TRAIN XGBOOST
    # ========================================================

    model = train_model(
        train_signals,
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
        f"\nSaved model:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    # ========================================================
    # TRAINING XGBOOST PREDICTIONS
    # ========================================================

    train_probabilities = (
        predict_probabilities(
            model,
            train_signals,
            feature_names,
        )
    )

    train_trade_mask = (
        np.isfinite(
            train_probabilities
        )
        &
        (
            train_probabilities
            >= PREDICTION_THRESHOLD
        )
    )

    # ========================================================
    # VALIDATION XGBOOST PREDICTIONS
    # ========================================================

    validation_probabilities = (
        predict_probabilities(
            model,
            validation_signals,
            feature_names,
        )
    )

    validation_trade_mask = (
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
    # BUILD FOUR EQUITY CURVES
    # ========================================================

    train_raw = build_equity_curve(
        train_signals
    )

    train_xgboost = build_equity_curve(
        train_signals,
        train_trade_mask,
    )

    validation_raw = build_equity_curve(
        validation_signals
    )

    validation_xgboost = build_equity_curve(
        validation_signals,
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
        "Train XGBoost",
        train_xgboost,
    )

    print_curve(
        "Validation Raw EMA",
        validation_raw,
    )

    print_curve(
        "Validation XGBoost",
        validation_xgboost,
    )

    # ========================================================
    # FILTER SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "XGBOOST FILTER"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPrediction threshold: "
        f"{PREDICTION_THRESHOLD:.2f}"
    )

    print(
        f"\nTraining:"
    )

    print(
        f"  EMA signals: "
        f"{len(train_signals):,}"
    )

    print(
        f"  Accepted: "
        f"{np.sum(train_trade_mask):,}"
    )

    print(
        f"  Rejected: "
        f"{len(train_trade_mask) - np.sum(train_trade_mask):,}"
    )

    print(
        f"\nValidation:"
    )

    print(
        f"  EMA signals: "
        f"{len(validation_signals):,}"
    )

    print(
        f"  Accepted: "
        f"{np.sum(validation_trade_mask):,}"
    )

    print(
        f"  Rejected: "
        f"{len(validation_trade_mask) - np.sum(validation_trade_mask):,}"
    )

    # ========================================================
    # SAVE FOUR CURVES
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
        "train_signals": train_signals,
        "validation_signals": validation_signals,
        "model": model,
        "train_raw": train_raw,
        "train_xgboost": train_xgboost,
        "validation_raw": validation_raw,
        "validation_xgboost": validation_xgboost,
    }


if __name__ == "__main__":
    main()