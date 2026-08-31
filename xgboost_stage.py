import io
import json
import os
import traceback

import boto3
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# R2 CONFIGURATION
# ============================================================

R2_ACCESS_KEY_ID = "00e18b0c16ecb3395cd6f7c8e0eb3554"
R2_SECRET_ACCESS_KEY = "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"
R2_ENDPOINT_URL = "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"

R2_BUCKET_NAME = "stocks-data"


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL = "DELL"

# ------------------------------------------------------------
# Which RR strategy are we training on?
# ------------------------------------------------------------

TRADE_TYPE = "long"

STOP_LOSS_PERCENTAGE = 1.0

RISK_REWARD_RATIO = 2.0


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# Maximum number of boosting trees.
#
# Early stopping may stop before this number.
#
# 20 was requested.
# ============================================================

N_ESTIMATORS = 20


# ------------------------------------------------------------
# Early stopping
# ------------------------------------------------------------

EARLY_STOPPING_ROUNDS = 5


# ------------------------------------------------------------
# Chronological validation split
#
# First 70% = training
# Last 30%  = validation
# ------------------------------------------------------------

TRAIN_PERCENTAGE = 0.70


# ------------------------------------------------------------
# Classification threshold
#
# XGBoost probability >= this value means:
#
#     TAKE TRADE
#
# Otherwise:
#
#     DON'T TRADE
# ------------------------------------------------------------

PREDICTION_THRESHOLD = 0.50


# ------------------------------------------------------------
# Random seed
# ------------------------------------------------------------

RANDOM_STATE = 42


# ============================================================
# INPUT / RR PATHS
# ============================================================

INPUT_PATH = "input"

RISK_REWARD_PATH = "riskreward"


# ============================================================
# OUTPUT PATHS
# ============================================================

XGBOOST_PATH = "xgboost"

MODEL_PATH = (
    f"{XGBOOST_PATH}/models"
)

DATASET_PATH = (
    f"{XGBOOST_PATH}/datasets"
)

IMPORTANCE_PATH = (
    f"{XGBOOST_PATH}/feature_importance"
)

STATISTICS_PATH = (
    f"{XGBOOST_PATH}/statistics"
)


# ============================================================
# LOGGING
# ============================================================

def log(message):

    print(
        f"[XGBOOST] {message}",
        flush=True,
    )


# ============================================================
# R2 CLIENT
# ============================================================

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
)


# ============================================================
# R2 HELPERS
# ============================================================

def download_bytes(key):

    response = s3.get_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
    )

    return response["Body"].read()


def upload_bytes(
    key,
    data,
    content_type=None,
):

    kwargs = {
        "Bucket": R2_BUCKET_NAME,
        "Key": key,
        "Body": data,
    }

    if content_type is not None:

        kwargs[
            "ContentType"
        ] = content_type

    s3.put_object(
        **kwargs
    )


def upload_dataframe(
    dataframe,
    key,
):

    buffer = io.BytesIO()

    dataframe.to_parquet(
        buffer,
        index=False,
    )

    buffer.seek(0)

    upload_bytes(
        key=key,
        data=buffer.getvalue(),
        content_type="application/octet-stream",
    )


def upload_json(
    data,
    key,
):

    text = json.dumps(
        data,
        indent=4,
        default=str,
    )

    upload_bytes(
        key=key,
        data=text.encode("utf-8"),
        content_type="application/json",
    )


# ============================================================
# DOWNLOAD INPUT DATA
# ============================================================

def download_input_data(
    symbol,
):

    key = (
        f"{INPUT_PATH}/"
        f"{symbol}.parquet"
    )

    log(
        f"Downloading input data | "
        f"key={key}"
    )

    raw = download_bytes(
        key
    )

    dataframe = pd.read_parquet(
        io.BytesIO(raw)
    )

    log(
        f"Input data downloaded | "
        f"rows={len(dataframe):,} | "
        f"columns={len(dataframe.columns):,}"
    )

    return dataframe


# ============================================================
# DOWNLOAD RISK REWARD DATA
# ============================================================

def download_risk_reward_data(
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
):

    key = (
        f"{RISK_REWARD_PATH}/"
        f"{symbol}/"
        f"{trade_type}/"
        f"{stop_loss_percentage}/"
        f"{risk_reward_ratio}.parquet"
    )

    log(
        f"Downloading RR data | "
        f"key={key}"
    )

    raw = download_bytes(
        key
    )

    dataframe = pd.read_parquet(
        io.BytesIO(raw)
    )

    log(
        f"RR data downloaded | "
        f"rows={len(dataframe):,}"
    )

    return dataframe


# ============================================================
# TIMESTAMP MATCHING
# ============================================================

def match_rr_to_input_rows(
    rr_timestamps,
    input_timestamps,
):
    """
    Match each RR entry timestamp to the latest
    input timestamp inside the corresponding
    5-minute candle.

    RR timestamp represents the 5-minute candle CLOSE.

    Example:

        RR:
            10:05

        Input:
            10:00
            10:01
            10:02
            10:03
            10:04
            10:05

        Selected:
            10:05

    If 10:05 does not exist:

            10:00
            10:01
            10:03
            10:04

        Selected:
            10:04
    """

    rr_timestamps = pd.DatetimeIndex(
        rr_timestamps
    )

    input_timestamps = pd.DatetimeIndex(
        input_timestamps
    )

    result = np.full(
        len(rr_timestamps),
        -1,
        dtype=np.int64,
    )

    input_ns = (
        input_timestamps
        .asi8
    )

    rr_ns = (
        rr_timestamps
        .asi8
    )

    five_minutes_ns = (
        pd.Timedelta(
            minutes=5
        ).value
    )

    for rr_index, rr_end in enumerate(
        rr_ns
    ):

        rr_start = (
            rr_end
            - five_minutes_ns
        )

        right = np.searchsorted(
            input_ns,
            rr_end,
            side="right",
        )

        left = np.searchsorted(
            input_ns,
            rr_start,
            side="left",
        )

        if right > left:

            result[
                rr_index
            ] = right - 1

    return result


# ============================================================
# PREPARE INPUT DATA
# ============================================================

def prepare_input_data(
    input_data,
):
    """
    Prepare indicator data for XGBoost.

    The timestamp is retained for pairing,
    but is NOT used as an XGBoost feature.

    All other numeric indicator columns are
    candidates for features.
    """

    input_data = (
        input_data
        .copy()
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "timestamp" not in input_data.columns:

        raise ValueError(
            "Input data does not contain "
            "'timestamp' column."
        )

    input_data["timestamp"] = (
        pd.to_datetime(
            input_data[
                "timestamp"
            ],
            utc=True,
        )
        .dt.tz_convert(
            "America/New_York"
        )
    )

    # --------------------------------------------------------
    # Remove duplicate timestamps.
    # --------------------------------------------------------

    input_data = (
        input_data
        .drop_duplicates(
            subset=[
                "timestamp"
            ],
            keep="first",
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Identify feature columns.
    #
    # Only numeric columns are allowed.
    # --------------------------------------------------------

    feature_columns = []

    for column in input_data.columns:

        if column == "timestamp":
            continue

        if pd.api.types.is_numeric_dtype(
            input_data[column]
        ):

            feature_columns.append(
                column
            )

    if len(feature_columns) == 0:

        raise ValueError(
            "No numeric input features "
            "were found."
        )

    # --------------------------------------------------------
    # Convert features to float64.
    # --------------------------------------------------------

    for column in feature_columns:

        input_data[column] = pd.to_numeric(
            input_data[column],
            errors="coerce",
        )

    log(
        f"Input preparation completed | "
        f"features={len(feature_columns):,}"
    )

    return (
        input_data,
        feature_columns,
    )


# ============================================================
# PREPARE RR DATA
# ============================================================

def prepare_rr_data(
    risk_reward_data,
):
    """
    Prepare RR data.

    The target is:

        trade_return_percent > 0

            1 = profitable / take trade

            0 = non-profitable / don't trade
    """

    risk_reward_data = (
        risk_reward_data
        .copy()
    )

    required_columns = [
        "start_timestamp",
        "trade_return_percent",
        "time_elapsed_minutes",
    ]

    for column in required_columns:

        if column not in risk_reward_data.columns:

            raise ValueError(
                f"RR data missing required "
                f"column: {column}"
            )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    risk_reward_data[
        "start_timestamp"
    ] = (
        pd.to_datetime(
            risk_reward_data[
                "start_timestamp"
            ],
            utc=True,
        )
        .dt.tz_convert(
            "America/New_York"
        )
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    risk_reward_data[
        "trade_return_percent"
    ] = pd.to_numeric(
        risk_reward_data[
            "trade_return_percent"
        ],
        errors="coerce",
    )

    risk_reward_data[
        "time_elapsed_minutes"
    ] = pd.to_numeric(
        risk_reward_data[
            "time_elapsed_minutes"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    risk_reward_data = (
        risk_reward_data
        .sort_values(
            "start_timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # TARGET
    #
    # Positive return = YES
    # Zero/negative = NO
    # --------------------------------------------------------

    risk_reward_data[
        "target"
    ] = (
        risk_reward_data[
            "trade_return_percent"
        ]
        > 0.0
    ).astype(
        np.int8
    )

    return risk_reward_data


# ============================================================
# BUILD XGBOOST DATASET
# ============================================================

def build_xgboost_dataset(
    input_data,
    risk_reward_data,
    feature_columns,
):
    """
    Pair RR trades with their corresponding
    indicator row.

    Output:

        timestamp
        all indicator features
        trade_return_percent
        time_elapsed_minutes
        target
    """

    # ========================================================
    # MATCH
    # ========================================================

    input_row_indices = (
        match_rr_to_input_rows(
            rr_timestamps=(
                risk_reward_data[
                    "start_timestamp"
                ]
            ),
            input_timestamps=(
                input_data[
                    "timestamp"
                ]
            ),
        )
    )

    # ========================================================
    # KEEP ONLY SUCCESSFULLY MATCHED ROWS
    # ========================================================

    valid_match = (
        input_row_indices >= 0
    )

    matched_count = int(
        np.sum(valid_match)
    )

    log(
        f"RR/Input matching | "
        f"RR rows={len(risk_reward_data):,} | "
        f"matched={matched_count:,} | "
        f"unmatched="
        f"{len(risk_reward_data) - matched_count:,}"
    )

    if matched_count == 0:

        raise ValueError(
            "No RR rows could be matched "
            "to input rows."
        )

    rr = (
        risk_reward_data.loc[
            valid_match
        ]
        .reset_index(
            drop=True
        )
    )

    input_indices = (
        input_row_indices[
            valid_match
        ]
    )

    # ========================================================
    # GET FEATURES
    # ========================================================

    X = (
        input_data[
            feature_columns
        ]
        .iloc[
            input_indices
        ]
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # BUILD DATASET
    # ========================================================

    dataset = X.copy()

    dataset.insert(
        0,
        "timestamp",
        rr[
            "start_timestamp"
        ].reset_index(
            drop=True
        ),
    )

    dataset[
        "trade_return_percent"
    ] = (
        rr[
            "trade_return_percent"
        ]
        .reset_index(
            drop=True
        )
    )

    dataset[
        "time_elapsed_minutes"
    ] = (
        rr[
            "time_elapsed_minutes"
        ]
        .reset_index(
            drop=True
        )
    )

    dataset[
        "target"
    ] = (
        rr[
            "target"
        ]
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # REMOVE INVALID FEATURE ROWS
    # ========================================================

    feature_values = (
        dataset[
            feature_columns
        ]
    )

    valid_features = np.all(
        np.isfinite(
            feature_values.to_numpy(
                dtype=np.float64
            )
        ),
        axis=1,
    )

    before = len(dataset)

    dataset = (
        dataset.loc[
            valid_features
        ]
        .reset_index(
            drop=True
        )
    )

    removed = (
        before
        - len(dataset)
    )

    log(
        f"Invalid feature rows removed | "
        f"rows={removed:,}"
    )

    # ========================================================
    # FINAL SORT
    # ========================================================

    dataset = (
        dataset
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return dataset


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_dataset(
    dataset,
):
    """
    Chronological split.

    First 70%:
        training

    Last 30%:
        validation
    """

    split_index = int(
        len(dataset)
        * TRAIN_PERCENTAGE
    )

    if split_index <= 0:

        raise ValueError(
            "Training dataset is empty."
        )

    if split_index >= len(dataset):

        raise ValueError(
            "Validation dataset is empty."
        )

    train_data = (
        dataset
        .iloc[
            :split_index
        ]
        .reset_index(
            drop=True
        )
    )

    validation_data = (
        dataset
        .iloc[
            split_index:
        ]
        .reset_index(
            drop=True
        )
    )

    log(
        f"Chronological split | "
        f"train={len(train_data):,} | "
        f"validation={len(validation_data):,}"
    )

    log(
        f"Training period | "
        f"{train_data['timestamp'].iloc[0]} "
        f"-> "
        f"{train_data['timestamp'].iloc[-1]}"
    )

    log(
        f"Validation period | "
        f"{validation_data['timestamp'].iloc[0]} "
        f"-> "
        f"{validation_data['timestamp'].iloc[-1]}"
    )

    return (
        train_data,
        validation_data,
    )


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_xgboost(
    train_data,
    validation_data,
    feature_columns,
):
    """
    Train XGBoost binary classifier.

    XGBoost learns the split points itself.

    No state grid is created.
    """

    X_train = (
        train_data[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_train = (
        train_data[
            "target"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    X_validation = (
        validation_data[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_validation = (
        validation_data[
            "target"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    log(
        "Starting XGBoost training"
    )

    log(
        f"Features={len(feature_columns):,}"
    )

    log(
        f"Training rows={len(X_train):,}"
    )

    log(
        f"Validation rows={len(X_validation):,}"
    )

    # ========================================================
    # CLASS BALANCE
    # ========================================================

    positive_count = int(
        np.sum(y_train == 1)
    )

    negative_count = int(
        np.sum(y_train == 0)
    )

    if positive_count == 0:

        raise ValueError(
            "Training data contains "
            "zero positive examples."
        )

    if negative_count == 0:

        raise ValueError(
            "Training data contains "
            "zero negative examples."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    log(
        f"Training positives={positive_count:,}"
    )

    log(
        f"Training negatives={negative_count:,}"
    )

    log(
        f"scale_pos_weight={scale_pos_weight:.4f}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = XGBClassifier(
        objective="binary:logistic",

        n_estimators=N_ESTIMATORS,

        # ----------------------------------------------------
        # Let early stopping determine how deep training
        # should go in terms of number of trees.
        # ----------------------------------------------------

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        min_child_weight=1,

        reg_alpha=0.0,

        reg_lambda=1.0,

        gamma=0.0,

        scale_pos_weight=scale_pos_weight,

        eval_metric="logloss",

        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),

        tree_method="hist",

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    # ========================================================
    # FIT
    # ========================================================

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=True,
    )

    log(
        "XGBoost training completed"
    )

    if hasattr(
        model,
        "best_iteration",
    ):

        log(
            f"Best iteration="
            f"{model.best_iteration}"
        )

    if hasattr(
        model,
        "best_score",
    ):

        log(
            f"Best validation score="
            f"{model.best_score}"
        )

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    validation_data,
    feature_columns,
):
    """
    Evaluate the model on chronological validation data.
    """

    X_validation = (
        validation_data[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_validation = (
        validation_data[
            "target"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )[
            :,
            1
        ]
    )

    predictions = (
        probabilities
        >= PREDICTION_THRESHOLD
    ).astype(
        np.int8
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    try:

        auc = roc_auc_score(
            y_validation,
            probabilities,
        )

    except ValueError:

        auc = None

    matrix = confusion_matrix(
        y_validation,
        predictions,
    )

    # ========================================================
    # TRADE SELECTION STATISTICS
    # ========================================================

    selected = (
        predictions == 1
    )

    selected_count = int(
        np.sum(selected)
    )

    selected_returns = (
        validation_data[
            "trade_return_percent"
        ]
        .to_numpy(
            dtype=np.float64
        )[
            selected
        ]
    )

    if selected_count > 0:

        selected_win_rate = float(
            np.mean(
                selected_returns > 0.0
            )
        )

        selected_average_return = float(
            np.mean(
                selected_returns
            )
        )

        selected_total_return = float(
            np.sum(
                selected_returns
            )
        )

    else:

        selected_win_rate = 0.0

        selected_average_return = 0.0

        selected_total_return = 0.0

    # ========================================================
    # RESULTS
    # ========================================================

    results = {
        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "roc_auc": (
            None
            if auc is None
            else float(auc)
        ),

        "confusion_matrix": (
            matrix.tolist()
        ),

        "validation_rows": int(
            len(validation_data)
        ),

        "predicted_take_trades": (
            selected_count
        ),

        "predicted_dont_trade": int(
            len(validation_data)
            - selected_count
        ),

        "predicted_trade_percentage": float(
            selected_count
            / len(validation_data)
            * 100.0
        ),

        "selected_trade_win_rate": (
            selected_win_rate
        ),

        "selected_trade_average_return_percent": (
            selected_average_return
        ),

        "selected_trade_total_return_percent": (
            selected_total_return
        ),

        "prediction_threshold": (
            PREDICTION_THRESHOLD
        ),
    }

    return results


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(
    model,
    feature_columns,
):
    """
    Calculate feature usefulness using XGBoost gain.

    Gain represents the average improvement in the
    objective from splits using the feature.

    Both raw gain and normalized gain are saved.
    """

    booster = (
        model.get_booster()
    )

    gain_scores = (
        booster.get_score(
            importance_type="gain"
        )
    )

    weight_scores = (
        booster.get_score(
            importance_type="weight"
        )
    )

    cover_scores = (
        booster.get_score(
            importance_type="cover"
        )
    )

    total_gain_scores = (
        booster.get_score(
            importance_type="total_gain"
        )
    )

    rows = []

    for feature in feature_columns:

        gain = float(
            gain_scores.get(
                feature,
                0.0,
            )
        )

        weight = float(
            weight_scores.get(
                feature,
                0.0,
            )
        )

        cover = float(
            cover_scores.get(
                feature,
                0.0,
            )
        )

        total_gain = float(
            total_gain_scores.get(
                feature,
                0.0,
            )
        )

        rows.append(
            {
                "feature": feature,

                "gain": gain,

                "total_gain": total_gain,

                "weight": weight,

                "cover": cover,
            }
        )

    importance = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Normalize gain.
    # --------------------------------------------------------

    total_gain = (
        importance[
            "gain"
        ].sum()
    )

    if total_gain > 0.0:

        importance[
            "gain_percentage"
        ] = (
            importance[
                "gain"
            ]
            / total_gain
            * 100.0
        )

    else:

        importance[
            "gain_percentage"
        ] = 0.0

    importance = (
        importance
        .sort_values(
            "gain_percentage",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance[
        "rank"
    ] = (
        np.arange(
            len(importance)
        )
        + 1
    )

    return importance


# ============================================================
# DATASET STATISTICS
# ============================================================

def calculate_dataset_statistics(
    dataset,
    train_data,
    validation_data,
    feature_columns,
):
    """
    Produce useful information about the training data.
    """

    total = len(dataset)

    positive = int(
        np.sum(
            dataset[
                "target"
            ].to_numpy()
            == 1
        )
    )

    negative = int(
        np.sum(
            dataset[
                "target"
            ].to_numpy()
            == 0
        )
    )

    statistics = {
        "symbol": SYMBOL,

        "trade_type": TRADE_TYPE,

        "stop_loss_percentage": (
            STOP_LOSS_PERCENTAGE
        ),

        "risk_reward_ratio": (
            RISK_REWARD_RATIO
        ),

        "total_rows": total,

        "training_rows": len(
            train_data
        ),

        "validation_rows": len(
            validation_data
        ),

        "feature_count": len(
            feature_columns
        ),

        "positive_rows": positive,

        "negative_rows": negative,

        "positive_percentage": (
            positive
            / total
            * 100.0
            if total > 0
            else 0.0
        ),

        "negative_percentage": (
            negative
            / total
            * 100.0
            if total > 0
            else 0.0
        ),

        "first_timestamp": str(
            dataset[
                "timestamp"
            ].iloc[0]
        ),

        "last_timestamp": str(
            dataset[
                "timestamp"
            ].iloc[-1]
        ),
    }

    return statistics


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
):
    """
    Save XGBoost model directly to R2.
    """

    filename = (
        f"{symbol}_"
        f"{trade_type}_"
        f"sl_{stop_loss_percentage}_"
        f"rr_{risk_reward_ratio}.json"
    )

    key = (
        f"{MODEL_PATH}/"
        f"{filename}"
    )

    # --------------------------------------------------------
    # XGBoost saves to a local path.
    # --------------------------------------------------------

    temporary_path = (
        "__xgboost_temp_model.json"
    )

    model.save_model(
        temporary_path
    )

    with open(
        temporary_path,
        "rb",
    ) as file:

        model_bytes = (
            file.read()
        )

    upload_bytes(
        key=key,
        data=model_bytes,
        content_type="application/json",
    )

    os.remove(
        temporary_path
    )

    log(
        f"Model saved | "
        f"key={key}"
    )

    return key


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    importance,
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
):

    filename = (
        f"{symbol}_"
        f"{trade_type}_"
        f"sl_{stop_loss_percentage}_"
        f"rr_{risk_reward_ratio}.parquet"
    )

    key = (
        f"{IMPORTANCE_PATH}/"
        f"{filename}"
    )

    upload_dataframe(
        importance,
        key,
    )

    log(
        f"Feature importance saved | "
        f"key={key}"
    )

    return key


# ============================================================
# SAVE PREPARED DATASET
# ============================================================

def save_prepared_dataset(
    dataset,
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
):

    filename = (
        f"{symbol}_"
        f"{trade_type}_"
        f"sl_{stop_loss_percentage}_"
        f"rr_{risk_reward_ratio}.parquet"
    )

    key = (
        f"{DATASET_PATH}/"
        f"{filename}"
    )

    upload_dataframe(
        dataset,
        key,
    )

    log(
        f"Prepared dataset saved | "
        f"key={key}"
    )

    return key


# ============================================================
# SAVE STATISTICS
# ============================================================

def save_statistics(
    statistics,
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
):

    filename = (
        f"{symbol}_"
        f"{trade_type}_"
        f"sl_{stop_loss_percentage}_"
        f"rr_{risk_reward_ratio}.json"
    )

    key = (
        f"{STATISTICS_PATH}/"
        f"{filename}"
    )

    upload_json(
        statistics,
        key,
    )

    log(
        f"Statistics saved | "
        f"key={key}"
    )

    return key


# ============================================================
# SAVE FEATURE LIST
# ============================================================

def save_feature_configuration(
    feature_columns,
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
):

    data = {
        "symbol": symbol,

        "trade_type": trade_type,

        "stop_loss_percentage": (
            stop_loss_percentage
        ),

        "risk_reward_ratio": (
            risk_reward_ratio
        ),

        "features": feature_columns,

        "feature_count": len(
            feature_columns
        ),

        "target": (
            "trade_return_percent > 0"
        ),

        "prediction_threshold": (
            PREDICTION_THRESHOLD
        ),

        "n_estimators": (
            N_ESTIMATORS
        ),

        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
    }

    filename = (
        f"{symbol}_"
        f"{trade_type}_"
        f"sl_{stop_loss_percentage}_"
        f"rr_{risk_reward_ratio}.json"
    )

    key = (
        f"{MODEL_PATH}/"
        f"{filename}.features.json"
    )

    upload_json(
        data,
        key,
    )

    log(
        f"Feature configuration saved | "
        f"key={key}"
    )

    return key


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        log(
            "=================================================="
        )

        log(
            "XGBOOST STAGE STARTED"
        )

        log(
            "=================================================="
        )

        log(
            f"Symbol={SYMBOL}"
        )

        log(
            f"Trade type={TRADE_TYPE}"
        )

        log(
            f"SL={STOP_LOSS_PERCENTAGE}"
        )

        log(
            f"RR={RISK_REWARD_RATIO}"
        )

        # ====================================================
        # DOWNLOAD INPUT
        # ====================================================

        input_data = (
            download_input_data(
                SYMBOL
            )
        )

        # ====================================================
        # DOWNLOAD RR
        # ====================================================

        risk_reward_data = (
            download_risk_reward_data(
                symbol=SYMBOL,
                trade_type=TRADE_TYPE,
                stop_loss_percentage=(
                    STOP_LOSS_PERCENTAGE
                ),
                risk_reward_ratio=(
                    RISK_REWARD_RATIO
                ),
            )
        )

        # ====================================================
        # PREPARE INPUT
        # ====================================================

        (
            input_data,
            feature_columns,
        ) = prepare_input_data(
            input_data
        )

        # ====================================================
        # PREPARE RR
        # ====================================================

        risk_reward_data = (
            prepare_rr_data(
                risk_reward_data
            )
        )

        # ====================================================
        # BUILD DATASET
        # ====================================================

        dataset = (
            build_xgboost_dataset(
                input_data=input_data,
                risk_reward_data=(
                    risk_reward_data
                ),
                feature_columns=(
                    feature_columns
                ),
            )
        )

        log(
            f"Final XGBoost dataset | "
            f"rows={len(dataset):,} | "
            f"features={len(feature_columns):,}"
        )

        # ====================================================
        # SPLIT
        # ====================================================

        (
            train_data,
            validation_data,
        ) = split_dataset(
            dataset
        )

        # ====================================================
        # DATASET STATISTICS
        # ====================================================

        dataset_statistics = (
            calculate_dataset_statistics(
                dataset=dataset,
                train_data=train_data,
                validation_data=(
                    validation_data
                ),
                feature_columns=(
                    feature_columns
                ),
            )
        )

        # ====================================================
        # TRAIN
        # ====================================================

        model = (
            train_xgboost(
                train_data=train_data,
                validation_data=(
                    validation_data
                ),
                feature_columns=(
                    feature_columns
                ),
            )
        )

        # ====================================================
        # EVALUATE
        # ====================================================

        evaluation_statistics = (
            evaluate_model(
                model=model,
                validation_data=(
                    validation_data
                ),
                feature_columns=(
                    feature_columns
                ),
            )
        )

        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        feature_importance = (
            calculate_feature_importance(
                model=model,
                feature_columns=(
                    feature_columns
                ),
            )
        )

        # ====================================================
        # PRINT TOP FEATURES
        # ====================================================

        log(
            "=================================================="
        )

        log(
            "TOP FEATURE IMPORTANCE"
        )

        log(
            "=================================================="
        )

        for _, row in (
            feature_importance
            .head(20)
            .iterrows()
        ):

            log(
                f"{int(row['rank']):>3} | "
                f"{row['feature']:<50} | "
                f"gain={row['gain_percentage']:.4f}%"
            )

        # ====================================================
        # COMBINE STATISTICS
        # ====================================================

        final_statistics = {
            **dataset_statistics,

            "model": {
                "n_estimators": (
                    N_ESTIMATORS
                ),

                "early_stopping_rounds": (
                    EARLY_STOPPING_ROUNDS
                ),

                "best_iteration": (
                    int(
                        model.best_iteration
                    )
                    if hasattr(
                        model,
                        "best_iteration",
                    )
                    else None
                ),

                "best_score": (
                    float(
                        model.best_score
                    )
                    if hasattr(
                        model,
                        "best_score",
                    )
                    else None
                ),
            },

            "validation": (
                evaluation_statistics
            ),
        }

        # ====================================================
        # SAVE PREPARED DATA
        # ====================================================

        save_prepared_dataset(
            dataset=dataset,
            symbol=SYMBOL,
            trade_type=TRADE_TYPE,
            stop_loss_percentage=(
                STOP_LOSS_PERCENTAGE
            ),
            risk_reward_ratio=(
                RISK_REWARD_RATIO
            ),
        )

        # ====================================================
        # SAVE MODEL
        # ====================================================

        save_model(
            model=model,
            symbol=SYMBOL,
            trade_type=TRADE_TYPE,
            stop_loss_percentage=(
                STOP_LOSS_PERCENTAGE
            ),
            risk_reward_ratio=(
                RISK_REWARD_RATIO
            ),
        )

        # ====================================================
        # SAVE IMPORTANCE
        # ====================================================

        save_feature_importance(
            importance=(
                feature_importance
            ),
            symbol=SYMBOL,
            trade_type=TRADE_TYPE,
            stop_loss_percentage=(
                STOP_LOSS_PERCENTAGE
            ),
            risk_reward_ratio=(
                RISK_REWARD_RATIO
            ),
        )

        # ====================================================
        # SAVE STATISTICS
        # ====================================================

        save_statistics(
            statistics=(
                final_statistics
            ),
            symbol=SYMBOL,
            trade_type=TRADE_TYPE,
            stop_loss_percentage=(
                STOP_LOSS_PERCENTAGE
            ),
            risk_reward_ratio=(
                RISK_REWARD_RATIO
            ),
        )

        # ====================================================
        # SAVE FEATURE CONFIGURATION
        # ====================================================

        save_feature_configuration(
            feature_columns=(
                feature_columns
            ),
            symbol=SYMBOL,
            trade_type=TRADE_TYPE,
            stop_loss_percentage=(
                STOP_LOSS_PERCENTAGE
            ),
            risk_reward_ratio=(
                RISK_REWARD_RATIO
            ),
        )

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        log(
            "=================================================="
        )

        log(
            "XGBOOST TRAINING COMPLETE"
        )

        log(
            "=================================================="
        )

        log(
            f"Rows={len(dataset):,}"
        )

        log(
            f"Features={len(feature_columns):,}"
        )

        log(
            f"Validation accuracy="
            f"{evaluation_statistics['accuracy']:.6f}"
        )

        log(
            f"Validation precision="
            f"{evaluation_statistics['precision']:.6f}"
        )

        log(
            f"Validation recall="
            f"{evaluation_statistics['recall']:.6f}"
        )

        log(
            f"Validation F1="
            f"{evaluation_statistics['f1']:.6f}"
        )

        if (
            evaluation_statistics[
                "roc_auc"
            ]
            is not None
        ):

            log(
                f"Validation ROC AUC="
                f"{evaluation_statistics['roc_auc']:.6f}"
            )

        log(
            f"Predicted trades="
            f"{evaluation_statistics['predicted_take_trades']:,}"
        )

        log(
            f"Predicted trade win rate="
            f"{evaluation_statistics['selected_trade_win_rate']:.6f}"
        )

        log(
            f"Average selected trade return="
            f"{evaluation_statistics['selected_trade_average_return_percent']:.6f}%"
        )

        log(
            "=================================================="
        )

    except Exception as error:

        log(
            "XGBOOST STAGE FAILED"
        )

        log(
            str(error)
        )

        traceback.print_exc()

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
