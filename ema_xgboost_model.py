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


# ============================================================
# CREATE R2 CLIENT
# ============================================================

def create_r2_client():
    """
    Create an S3-compatible Cloudflare R2 client.
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

def download_parquet(client, path):
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
# DOWNLOAD TRAINING DATA
# ============================================================

def download_training_data(client):
    """
    Download the training EMA crossover data.
    """

    print(
        f"Downloading training data:"
    )

    print(
        f"  {TRAIN_SIGNALS_PATH}"
    )

    train_signal_data = download_parquet(
        client,
        TRAIN_SIGNALS_PATH,
    )

    print(
        f"Training signals: "
        f"{len(train_signal_data):,}"
    )

    return train_signal_data


# ============================================================
# GET FEATURE NAMES
# ============================================================

def get_feature_names(data):
    """
    Select exactly the first 27 feature columns.

    Column 0 is timestamp.

    Columns 1:28 are the 27 model features.
    """

    if len(data.columns) < FEATURE_END_COLUMN:
        raise ValueError(
            "Training data does not contain "
            "at least 28 columns."
        )

    feature_names = list(
        data.columns[
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
# BUILD X / Y
# ============================================================

def build_training_data(
    train_signal_data,
    feature_names,
):
    """
    Build the XGBoost feature matrix and target vector.

    Target:

        1 = trade_return_percent > 0
        0 = trade_return_percent <= 0
    """

    if len(train_signal_data) == 0:
        raise ValueError(
            "There are no training EMA crossover signals."
        )

    missing_features = [
        feature
        for feature in feature_names
        if feature not in train_signal_data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing training features: "
            + ", ".join(missing_features)
        )

    if "target" not in train_signal_data.columns:
        raise ValueError(
            "Training data does not contain 'target'."
        )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = (
        train_signal_data[feature_names]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y = (
        pd.to_numeric(
            train_signal_data["target"],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    valid_mask = (
        np.all(
            np.isfinite(X),
            axis=1,
        )
        &
        np.isfinite(y)
    )

    X = X[valid_mask]

    y = y[valid_mask]

    # Convert target to integer.
    y = y.astype(
        np.int8
    )

    if len(X) == 0:
        raise ValueError(
            "No valid training rows remain."
        )

    # --------------------------------------------------------
    # CHECK TARGET CLASSES
    # --------------------------------------------------------

    unique_targets = np.unique(y)

    if len(unique_targets) < 2:
        raise ValueError(
            "Training data contains only one target class. "
            "XGBoost requires both profitable and "
            "non-profitable examples."
        )

    return X, y


# ============================================================
# CREATE MODEL
# ============================================================

def create_xgboost_model():
    """
    Create a deliberately small XGBoost classifier.
    """

    model = XGBClassifier(
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

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_xgboost_model(
    train_signal_data,
    feature_names,
):
    """
    Train XGBoost using only the training period.
    """

    X, y = build_training_data(
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
        f"\nTraining rows: "
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

    print(
        "\nFitting model..."
    )

    model = create_xgboost_model()

    model.fit(
        X,
        y,
        verbose=False,
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # TRAINING ACCURACY
    # --------------------------------------------------------

    train_predictions = model.predict(X)

    train_accuracy = (
        np.mean(
            train_predictions == y
        )
        * 100.0
    )

    print(
        f"Training accuracy: "
        f"{train_accuracy:.2f}%"
    )

    return model


# ============================================================
# PREDICT PROFITABILITY
# ============================================================

def predict_profitability(
    model,
    signal_data,
    feature_names,
):
    """
    Return the model's probability that each EMA crossover
    will produce a positive RR return.

    This predicts profitability, NOT return magnitude.
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
# GET XGBOOST TRADE MASK
# ============================================================

def get_xgboost_trade_mask(
    model,
    signal_data,
    feature_names,
):
    """
    Determine which EMA crossover signals should actually
    be traded.

    A signal is traded when:

        predicted probability >= threshold
    """

    probabilities = predict_profitability(
        model,
        signal_data,
        feature_names,
    )

    trade_mask = (
        probabilities
        >= PREDICTION_THRESHOLD
    )

    return (
        trade_mask,
        probabilities,
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    client,
    model,
):
    """
    Save the trained XGBoost model directly to R2.
    """

    print(
        f"\nSaving model:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    # Save locally in memory first because XGBoost's
    # save_model expects a filename.
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        suffix=".json",
        delete=False,
    ) as temp_file:

        temp_path = temp_file.name

    try:

        model.save_model(
            temp_path
        )

        with open(
            temp_path,
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
            temp_path
        ):
            os.remove(
                temp_path
            )

    print(
        "Model saved."
    )


# ============================================================
# PRINT TARGET DISTRIBUTION
# ============================================================

def print_target_distribution(
    train_signal_data,
):
    """
    Print the profitable/non-profitable distribution.
    """

    target = (
        train_signal_data["target"]
        .to_numpy()
    )

    profitable = int(
        np.sum(target == 1)
    )

    non_profitable = int(
        np.sum(target == 0)
    )

    total = len(target)

    print(
        "\n" + "=" * 70
    )
    print(
        "TRAINING TARGET DISTRIBUTION"
    )
    print(
        "=" * 70
    )

    print(
        f"\nProfitable:     "
        f"{profitable:,}"
    )

    print(
        f"Not profitable: "
        f"{non_profitable:,}"
    )

    print(
        f"Total:          "
        f"{total:,}"
    )

    if total > 0:

        print(
            f"Profitable %:   "
            f"{profitable / total * 100:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EMA XGBOOST MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    client = create_r2_client()

    # --------------------------------------------------------
    # DOWNLOAD TRAINING SIGNALS
    # --------------------------------------------------------

    train_signal_data = (
        download_training_data(
            client
        )
    )

    # --------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------

    feature_names = get_feature_names(
        train_signal_data
    )

    print(
        f"\nUsing {len(feature_names)} features."
    )

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    print_target_distribution(
        train_signal_data
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = train_xgboost_model(
        train_signal_data,
        feature_names,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_model(
        client,
        model,
    )

    # --------------------------------------------------------
    # OPTIONAL LOCAL CHECK
    # --------------------------------------------------------

    probabilities = predict_profitability(
        model,
        train_signal_data,
        feature_names,
    )

    valid_probabilities = (
        probabilities[
            np.isfinite(probabilities)
        ]
    )

    if len(valid_probabilities) > 0:

        print(
            "\n" + "=" * 70
        )
        print(
            "MODEL PROBABILITY SUMMARY"
        )
        print(
            "=" * 70
        )

        print(
            f"\nMinimum probability: "
            f"{np.min(valid_probabilities):.4f}"
        )

        print(
            f"Maximum probability: "
            f"{np.max(valid_probabilities):.4f}"
        )

        print(
            f"Mean probability:    "
            f"{np.mean(valid_probabilities):.4f}"
        )

        print(
            f"Threshold:           "
            f"{PREDICTION_THRESHOLD:.2f}"
        )

        training_trade_mask = (
            valid_probabilities
            >= PREDICTION_THRESHOLD
        )

        print(
            f"Training signals passing "
            f"threshold: "
            f"{np.sum(training_trade_mask):,}"
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "XGBOOST MODEL COMPLETE"
    )
    print(
        "=" * 70
    )

    return {
        "model": model,
        "feature_names": feature_names,
        "train_signal_data": train_signal_data,
    }


if __name__ == "__main__":
    main()