import io
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

FEATURE_START_COLUMN = 1
FEATURE_END_COLUMN = 28

PREDICTION_THRESHOLD = 0.50


# ============================================================
# XGBOOST CONFIGURATION
# ============================================================

MAX_ESTIMATORS = 20
MAX_DEPTH = 3
LEARNING_RATE = 0.05

SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

RANDOM_STATE = 42
XGBOOST_DEVICE = "cuda"


# ============================================================
# R2 PATHS
# ============================================================

OUTPUT_ROOT = f"ema_xgboost/{SYMBOL}"

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
# CREATE R2 CLIENT
# ============================================================

def create_r2_client():
    """
    Create the Cloudflare R2 client.
    """

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )


# ============================================================
# DOWNLOAD PARQUET
# ============================================================

def download_parquet(
    client,
    path,
):
    """
    Download a parquet file from R2.
    """

    response = client.get_object(
        Bucket=R2_BUCKET_NAME,
        Key=path,
    )

    data = response["Body"].read()

    return pd.read_parquet(
        io.BytesIO(data)
    )


# ============================================================
# UPLOAD PARQUET
# ============================================================

def upload_parquet(
    client,
    dataframe,
    path,
):
    """
    Upload a DataFrame as parquet to R2.
    """

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
    """
    Download the trained XGBoost model from R2.
    """

    response = client.get_object(
        Bucket=R2_BUCKET_NAME,
        Key=MODEL_PATH,
    )

    model_bytes = response["Body"].read()

    model = XGBClassifier()

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        suffix=".json",
        delete=False,
    ) as temp_file:

        temp_path = temp_file.name
        temp_file.write(model_bytes)

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
# GET FEATURE NAMES
# ============================================================

def get_feature_names(
    signal_data,
):
    """
    Use exactly columns 1 through 27 as features.
    """

    if len(signal_data.columns) < FEATURE_END_COLUMN:
        raise ValueError(
            "Signal data does not contain "
            "at least 28 columns."
        )

    feature_names = list(
        signal_data.columns[
            FEATURE_START_COLUMN:
            FEATURE_END_COLUMN
        ]
    )

    if len(feature_names) != 27:
        raise ValueError(
            f"Expected 27 features, "
            f"got {len(feature_names)}."
        )

    return feature_names


# ============================================================
# PREDICT XGBOOST PROFITABILITY
# ============================================================

def predict_profitability(
    model,
    signal_data,
    feature_names,
):
    """
    Predict the probability that each EMA crossover
    produces a positive RR return.
    """

    if len(signal_data) == 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    X = (
        signal_data[feature_names]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    finite_mask = np.all(
        np.isfinite(X),
        axis=1,
    )

    probabilities = np.full(
        len(signal_data),
        np.nan,
        dtype=np.float32,
    )

    if np.any(finite_mask):

        probabilities[finite_mask] = (
            model.predict_proba(
                X[finite_mask]
            )[:, 1]
        )

    return probabilities


# ============================================================
# BUILD XGBOOST TRADE MASK
# ============================================================

def get_xgboost_trade_mask(
    model,
    signal_data,
    feature_names,
):
    """
    A trade is taken when the predicted probability
    is at least PREDICTION_THRESHOLD.
    """

    probabilities = predict_profitability(
        model,
        signal_data,
        feature_names,
    )

    trade_mask = (
        np.isfinite(probabilities)
        &
        (
            probabilities
            >= PREDICTION_THRESHOLD
        )
    )

    return (
        trade_mask,
        probabilities,
    )


# ============================================================
# BUILD EQUITY CURVE
# ============================================================

def build_equity_curve(
    signal_data,
    trade_mask=None,
):
    """
    Build an independently compounded equity curve.

    Starting equity:

        1.0

    Each selected trade changes equity by:

        equity *= 1 + trade_return_percent / 100

    If trade_mask is None, every EMA crossover signal
    is traded.

    If trade_mask is supplied, only True rows are traded.
    """

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
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # RAW STRATEGY
    # --------------------------------------------------------

    if trade_mask is None:

        selected = data.copy()

    # --------------------------------------------------------
    # FILTERED STRATEGY
    # --------------------------------------------------------

    else:

        trade_mask = np.asarray(
            trade_mask,
            dtype=bool,
        )

        if len(trade_mask) != len(data):
            raise ValueError(
                "trade_mask length does not match "
                "signal_data length."
            )

        selected = (
            data.loc[trade_mask]
            .copy()
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # NO TRADES
    # --------------------------------------------------------

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

    trade_returns = (
        pd.to_numeric(
            selected["trade_return_percent"],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    valid_returns = np.isfinite(
        trade_returns
    )

    selected = (
        selected
        .loc[valid_returns]
        .reset_index(drop=True)
    )

    trade_returns = trade_returns[
        valid_returns
    ]

    # --------------------------------------------------------
    # COMPOUND EQUITY
    # --------------------------------------------------------

    equity_multipliers = (
        1.0
        +
        trade_returns / 100.0
    )

    equity_values = np.cumprod(
        equity_multipliers,
        dtype=np.float64,
    )

    equity_curve = pd.DataFrame(
        {
            "timestamp": selected[
                "timestamp"
            ].to_numpy(),

            "trade_return_percent": trade_returns,

            "equity": equity_values,
        }
    )

    return equity_curve


# ============================================================
# RAW EMA STRATEGY
# ============================================================

def run_raw_ema_strategy(
    signal_data,
):
    """
    Trade every EMA crossover.
    """

    return build_equity_curve(
        signal_data,
        trade_mask=None,
    )


# ============================================================
# XGBOOST STRATEGY
# ============================================================

def run_xgboost_strategy(
    model,
    signal_data,
    feature_names,
):
    """
    Trade EMA crossovers only when XGBoost predicts
    that the trade has at least the configured
    profitability probability.
    """

    trade_mask, probabilities = (
        get_xgboost_trade_mask(
            model,
            signal_data,
            feature_names,
        )
    )

    equity_curve = build_equity_curve(
        signal_data,
        trade_mask=trade_mask,
    )

    return (
        equity_curve,
        trade_mask,
        probabilities,
    )


# ============================================================
# SUMMARIZE EQUITY CURVE
# ============================================================

def summarize_equity_curve(
    equity_curve,
):
    """
    Calculate basic statistics for one equity curve.
    """

    if len(equity_curve) == 0:

        return {
            "trades": 0,
            "start_equity": 1.0,
            "end_equity": 1.0,
            "total_return_percent": 0.0,
            "win_rate_percent": 0.0,
        }

    trade_returns = (
        equity_curve[
            "trade_return_percent"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    start_equity = 1.0

    end_equity = float(
        equity_curve[
            "equity"
        ].iloc[-1]
    )

    trades = len(
        trade_returns
    )

    winning_trades = np.sum(
        trade_returns > 0.0
    )

    win_rate = (
        winning_trades
        / trades
        * 100.0
    )

    total_return = (
        end_equity
        - start_equity
    ) * 100.0

    return {
        "trades": trades,
        "start_equity": start_equity,
        "end_equity": end_equity,
        "total_return_percent": total_return,
        "win_rate_percent": win_rate,
    }


# ============================================================
# PRINT CURVE SUMMARY
# ============================================================

def print_curve_summary(
    name,
    equity_curve,
):
    """
    Print start and end equity along with
    basic strategy statistics.
    """

    summary = summarize_equity_curve(
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
# SAVE ALL EQUITY CURVES
# ============================================================

def save_equity_curves(
    client,
    train_raw,
    train_xgboost,
    validation_raw,
    validation_xgboost,
):
    """
    Save each equity curve independently.
    """

    print(
        "\n" + "=" * 70
    )
    print(
        "SAVING EQUITY CURVES"
    )
    print(
        "=" * 70
    )

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

    for curve, path in curves:

        upload_parquet(
            client,
            curve,
            path,
        )

        print(
            f"Saved: {path}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EMA XGBOOST STRATEGY EVALUATION")
    print("=" * 70)

    client = create_r2_client()

    # --------------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------------

    print(
        "\nDownloading training signals..."
    )

    train_signal_data = download_parquet(
        client,
        TRAIN_SIGNALS_PATH,
    )

    print(
        f"Training signals: "
        f"{len(train_signal_data):,}"
    )

    print(
        "\nDownloading validation signals..."
    )

    validation_signal_data = download_parquet(
        client,
        VALIDATION_SIGNALS_PATH,
    )

    print(
        f"Validation signals: "
        f"{len(validation_signal_data):,}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_names = get_feature_names(
        train_signal_data
    )

    # --------------------------------------------------------
    # DOWNLOAD TRAINED MODEL
    # --------------------------------------------------------

    print(
        "\nDownloading trained XGBoost model..."
    )

    model = download_model(
        client
    )

    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # TRAINING RAW
    # --------------------------------------------------------

    print(
        "\nBuilding training raw EMA curve..."
    )

    train_raw = run_raw_ema_strategy(
        train_signal_data
    )

    # --------------------------------------------------------
    # TRAINING XGBOOST
    # --------------------------------------------------------

    print(
        "Building training XGBoost curve..."
    )

    (
        train_xgboost,
        train_trade_mask,
        train_probabilities,
    ) = run_xgboost_strategy(
        model,
        train_signal_data,
        feature_names,
    )

    # --------------------------------------------------------
    # VALIDATION RAW
    # --------------------------------------------------------

    print(
        "Building validation raw EMA curve..."
    )

    validation_raw = run_raw_ema_strategy(
        validation_signal_data
    )

    # --------------------------------------------------------
    # VALIDATION XGBOOST
    # --------------------------------------------------------

    print(
        "Building validation XGBoost curve..."
    )

    (
        validation_xgboost,
        validation_trade_mask,
        validation_probabilities,
    ) = run_xgboost_strategy(
        model,
        validation_signal_data,
        feature_names,
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "EQUITY CURVES"
    )
    print(
        "=" * 70
    )

    print_curve_summary(
        "Train Raw EMA",
        train_raw,
    )

    print_curve_summary(
        "Train XGBoost",
        train_xgboost,
    )

    print_curve_summary(
        "Validation Raw EMA",
        validation_raw,
    )

    print_curve_summary(
        "Validation XGBoost",
        validation_xgboost,
    )

    # --------------------------------------------------------
    # PRINT FILTER COUNTS
    # --------------------------------------------------------

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
        f"\nThreshold: "
        f"{PREDICTION_THRESHOLD:.2f}"
    )

    print(
        f"\nTraining:"
    )

    print(
        f"  EMA signals: "
        f"{len(train_signal_data):,}"
    )

    print(
        f"  Trades taken: "
        f"{np.sum(train_trade_mask):,}"
    )

    print(
        f"  Trades rejected: "
        f"{len(train_trade_mask) - np.sum(train_trade_mask):,}"
    )

    print(
        f"\nValidation:"
    )

    print(
        f"  EMA signals: "
        f"{len(validation_signal_data):,}"
    )

    print(
        f"  Trades taken: "
        f"{np.sum(validation_trade_mask):,}"
    )

    print(
        f"  Trades rejected: "
        f"{len(validation_trade_mask) - np.sum(validation_trade_mask):,}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_equity_curves(
        client,
        train_raw,
        train_xgboost,
        validation_raw,
        validation_xgboost,
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "STRATEGY EVALUATION COMPLETE"
    )
    print(
        "=" * 70
    )

    return {
        "train_raw": train_raw,
        "train_xgboost": train_xgboost,
        "validation_raw": validation_raw,
        "validation_xgboost": validation_xgboost,
        "train_trade_mask": train_trade_mask,
        "validation_trade_mask": validation_trade_mask,
        "train_probabilities": train_probabilities,
        "validation_probabilities": validation_probabilities,
    }


if __name__ == "__main__":
    main()