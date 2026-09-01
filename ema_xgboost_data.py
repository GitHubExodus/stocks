import io
import boto3
import numpy as np
import pandas as pd


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

TRAIN_END_DATE = "2024-12-31"
VALIDATION_START_DATE = "2025-01-01"

# First column is timestamp.
# Columns 1 through 27 inclusive = 27 features.
FEATURE_START_COLUMN = 1
FEATURE_END_COLUMN = 28

EMA_FAST_COLUMN = "ema_distance_9"
EMA_SLOW_COLUMN = "ema_distance_21"

RR_RETURN_COLUMN = "trade_return_percent"


# ============================================================
# NEW R2 STORAGE LOCATION
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


# ============================================================
# CREATE R2 CLIENT
# ============================================================

def create_r2_client():
    """
    Create an S3-compatible client for Cloudflare R2.
    """

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )


# ============================================================
# R2 DOWNLOAD
# ============================================================

def download_parquet(client, path):
    """
    Download a parquet file from R2 and return it as a DataFrame.
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
# R2 UPLOAD
# ============================================================

def upload_parquet(client, dataframe, path):
    """
    Save a DataFrame as parquet directly to R2.
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
# SOURCE PATHS
# ============================================================

def get_input_path():
    """
    Existing input data location.

    This is only where the source data is read from.
    The output is saved somewhere completely different.
    """

    return f"input/{SYMBOL}.parquet"


def get_risk_reward_path():
    """
    Existing risk/reward source location.
    """

    return (
        f"riskreward/"
        f"{SYMBOL}/"
        f"{TRADE_TYPE}/"
        f"sl_{float(STOP_LOSS_PERCENTAGE)}_"
        f"rr_{float(RISK_REWARD_RATIO)}.parquet"
    )


# ============================================================
# PREPARE INPUT DATA
# ============================================================

def prepare_input_data(input_data):
    """
    Prepare the 1-minute input data.

    The input timestamp is converted from UTC to
    America/New_York.

    Duplicate timestamps are removed.

    Data is sorted chronologically.
    """

    data = input_data.copy()

    if "timestamp" not in data.columns:
        raise ValueError(
            "Input data does not contain a 'timestamp' column."
        )

    data["timestamp"] = (
        pd.to_datetime(
            data["timestamp"],
            utc=True,
        )
        .dt.tz_convert(INPUT_TIMEZONE)
    )

    data = (
        data
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return data


# ============================================================
# PREPARE RISK/REWARD DATA
# ============================================================

def prepare_risk_reward_data(risk_reward_data):
    """
    Prepare the 5-minute risk/reward data.

    start_timestamp is converted from UTC to
    America/New_York.

    Data is sorted chronologically.
    """

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
        .dt.tz_convert(INPUT_TIMEZONE)
    )

    data = (
        data
        .sort_values("start_timestamp")
        .reset_index(drop=True)
    )

    return data


# ============================================================
# MATCH RR ROWS TO INPUT ROWS
# ============================================================

def match_rr_to_input_rows(
    input_data,
    risk_reward_data,
):
    """
    Match every RR timestamp to the latest available
    1-minute input row inside:

        (RR timestamp - 5 minutes, RR timestamp]

    The left boundary is exclusive.

    Missing 1-minute bars are allowed.

    We do NOT require exactly five 1-minute bars.

    If there are no input bars inside the window,
    the RR row receives no match.

    Returns:
        numpy array containing the input row index
        for each RR row, or -1 when no match exists.
    """

    input_timestamps = pd.DatetimeIndex(
        input_data["timestamp"]
    )

    rr_timestamps = pd.DatetimeIndex(
        risk_reward_data["start_timestamp"]
    )

    input_ns = input_timestamps.asi8
    rr_ns = rr_timestamps.asi8

    five_minutes = pd.Timedelta(
        minutes=5
    ).value

    matched_input_indices = np.full(
        len(rr_timestamps),
        -1,
        dtype=np.int64,
    )

    for i in range(len(rr_timestamps)):

        rr_end = rr_ns[i]

        rr_start = (
            rr_end -
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
        # This makes the left boundary exclusive.
        left = np.searchsorted(
            input_ns,
            rr_start,
            side="right",
        )

        if right > left:
            matched_input_indices[i] = right - 1

    return matched_input_indices


# ============================================================
# BUILD ALIGNED DATASET
# ============================================================

def build_aligned_dataset(input_data, risk_reward_data):
    """
    Build the dataset used by the EMA crossover + XGBoost system.

    XGBoost features:
        Only columns 2 through 28 whose names end exactly in
        "_3" or "_5".

    EMA signal:
        ema_distance_9
        ema_distance_21

    The EMA columns do NOT need to be XGBoost features.
    """

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    input_data = prepare_input_data(input_data)
    risk_reward_data = prepare_risk_reward_data(risk_reward_data)

    # --------------------------------------------------------
    # FIND XGBOOST FEATURES
    #
    # Python columns[1:28] means:
    #   column 2 through column 28
    # --------------------------------------------------------

    if len(input_data.columns) < 28:
        raise ValueError(
            f"Input data only has {len(input_data.columns)} columns. "
            "At least 28 columns are required."
        )

    candidate_columns = list(input_data.columns[1:28])

    feature_names = [
        column
        for column in candidate_columns
        if column.endswith("_3") or column.endswith("_5")
    ]

    if not feature_names:
        raise ValueError(
            "No XGBoost features ending in '_3' or '_5' "
            "were found between columns 2 and 28."
        )

    # --------------------------------------------------------
    # EMA COLUMNS
    #
    # These are NOT XGBoost features.
    # They are only used to generate the crossover signal.
    # --------------------------------------------------------

    if EMA_FAST_COLUMN not in input_data.columns:
        raise ValueError(
            f"{EMA_FAST_COLUMN} is missing from input data."
        )

    if EMA_SLOW_COLUMN not in input_data.columns:
        raise ValueError(
            f"{EMA_SLOW_COLUMN} is missing from input data."
        )

    # --------------------------------------------------------
    # CALCULATE EMA CROSSOVER ON THE FULL 1-MINUTE DATA
    #
    # This must happen BEFORE matching the RR rows.
    # Otherwise shift(1) would refer to the previous RR row
    # instead of the previous 1-minute bar.
    # --------------------------------------------------------

    fast = pd.to_numeric(
        input_data[EMA_FAST_COLUMN],
        errors="coerce",
    )

    slow = pd.to_numeric(
        input_data[EMA_SLOW_COLUMN],
        errors="coerce",
    )

    input_data["ema_crossover"] = (
        (fast.shift(1) <= slow.shift(1))
        & (fast > slow)
    )

    # --------------------------------------------------------
    # MATCH EACH RR BAR TO THE LATEST INPUT BAR
    # WITHIN THE PREVIOUS 5 MINUTES
    # --------------------------------------------------------

    matched_indices = match_rr_to_input_rows(
        input_data,
        risk_reward_data,
    )

    valid_matches = matched_indices >= 0

    if not np.any(valid_matches):
        raise ValueError(
            "No risk/reward rows could be matched to input data."
        )

    # --------------------------------------------------------
    # KEEP ONLY VALID MATCHES
    # --------------------------------------------------------

    rr = risk_reward_data.loc[
        valid_matches
    ].reset_index(drop=True)

    input_indices = matched_indices[valid_matches]

    input_rows = input_data.iloc[
        input_indices
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # BUILD ALIGNED DATASET
    # --------------------------------------------------------

    dataset = pd.DataFrame()

    # Timestamp
    dataset["timestamp"] = rr[
        "start_timestamp"
    ].to_numpy()

    # --------------------------------------------------------
    # XGBOOST FEATURES
    #
    # Only the selected _3 / _5 columns.
    # --------------------------------------------------------

    for feature_name in feature_names:
        dataset[feature_name] = input_rows[
            feature_name
        ].to_numpy()

    # --------------------------------------------------------
    # ACTUAL TRADE RETURN
    # --------------------------------------------------------

    dataset["trade_return_percent"] = pd.to_numeric(
        rr["trade_return_percent"],
        errors="coerce",
    ).to_numpy()

    # --------------------------------------------------------
    # EMA CROSSOVER SIGNAL
    # --------------------------------------------------------

    dataset["ema_crossover"] = input_rows[
        "ema_crossover"
    ].to_numpy()

    # --------------------------------------------------------
    # TARGET
    #
    # 1 = profitable trade
    # 0 = non-profitable trade
    # --------------------------------------------------------

    dataset["target"] = (
        dataset["trade_return_percent"] > 0.0
    ).astype(np.int8)

    # --------------------------------------------------------
    # CLEAN INVALID VALUES
    # --------------------------------------------------------

    numeric_columns = (
        feature_names
        + ["trade_return_percent"]
    )

    for column in numeric_columns:
        dataset[column] = pd.to_numeric(
            dataset[column],
            errors="coerce",
        )

    feature_values = dataset[
        feature_names
    ].to_numpy(dtype=np.float64)

    return_values = dataset[
        "trade_return_percent"
    ].to_numpy(dtype=np.float64)

    valid_numeric = (
        np.all(np.isfinite(feature_values), axis=1)
        & np.isfinite(return_values)
    )

    dataset = dataset.loc[
        valid_numeric
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # SORT BY TIMESTAMP
    # --------------------------------------------------------

    dataset = dataset.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # PRINT FEATURE INFORMATION
    # --------------------------------------------------------

    print()
    print("XGBoost features:")
    print(f"  Candidate columns: {len(candidate_columns)}")
    print(f"  Selected features: {len(feature_names)}")

    for feature_name in feature_names:
        print(f"    {feature_name}")

    print()
    print("EMA signal columns:")
    print(f"  Fast EMA: {EMA_FAST_COLUMN}")
    print(f"  Slow EMA: {EMA_SLOW_COLUMN}")

    print()
    print(f"Aligned rows: {len(dataset):,}")

    return dataset, feature_names

# ============================================================
# SPLIT TRAIN / VALIDATION
# ============================================================

def split_train_validation(dataset):
    """
    Training:
        Through December 31, 2024.

    Validation:
        January 1, 2025 onward.

    The split is chronological.
    """

    timestamps = dataset["timestamp"]

    train_mask = (
        timestamps
        <= pd.Timestamp(
            TRAIN_END_DATE,
            tz=INPUT_TIMEZONE,
        )
    )

    validation_mask = (
        timestamps
        >= pd.Timestamp(
            VALIDATION_START_DATE,
            tz=INPUT_TIMEZONE,
        )
    )

    train_data = (
        dataset.loc[train_mask]
        .reset_index(drop=True)
    )

    validation_data = (
        dataset.loc[validation_mask]
        .reset_index(drop=True)
    )

    return train_data, validation_data


# ============================================================
# GET EMA SIGNAL DATA
# ============================================================

def get_ema_signal_data(dataset):
    """
    Return only bars where the 9 EMA distance
    crosses above the 21 EMA distance.
    """

    signal_data = (
        dataset.loc[
            dataset["ema_crossover"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    return signal_data


# ============================================================
# DOWNLOAD SOURCE DATA
# ============================================================

def download_source_data(client):
    """
    Download the two source datasets directly from R2.
    """

    input_path = get_input_path()
    rr_path = get_risk_reward_path()

    print(
        f"\nDownloading input data:"
    )
    print(
        f"  {input_path}"
    )

    input_data = download_parquet(
        client,
        input_path,
    )

    print(
        f"Input rows: "
        f"{len(input_data):,}"
    )

    print(
        f"\nDownloading risk/reward data:"
    )
    print(
        f"  {rr_path}"
    )

    risk_reward_data = download_parquet(
        client,
        rr_path,
    )

    print(
        f"RR rows: "
        f"{len(risk_reward_data):,}"
    )

    return (
        input_data,
        risk_reward_data,
    )


# ============================================================
# SAVE PREPARED DATA
# ============================================================

def save_prepared_data(
    client,
    dataset,
    train_signal_data,
    validation_signal_data,
):
    """
    Save the prepared datasets to the new
    ema_xgboost folder.
    """

    print(
        "\nUploading prepared datasets..."
    )

    upload_parquet(
        client,
        dataset,
        ALIGNED_DATA_PATH,
    )

    print(
        f"Saved: {ALIGNED_DATA_PATH}"
    )

    upload_parquet(
        client,
        train_signal_data,
        TRAIN_SIGNALS_PATH,
    )

    print(
        f"Saved: {TRAIN_SIGNALS_PATH}"
    )

    upload_parquet(
        client,
        validation_signal_data,
        VALIDATION_SIGNALS_PATH,
    )

    print(
        f"Saved: {VALIDATION_SIGNALS_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EMA XGBOOST DATA PREPARATION")
    print("=" * 70)

    client = create_r2_client()

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    input_data, risk_reward_data = (
        download_source_data(client)
    )

    # --------------------------------------------------------
    # ALIGN
    # --------------------------------------------------------

    print(
        "\nBuilding aligned dataset..."
    )

    dataset, feature_names = (
        build_aligned_dataset(
            input_data,
            risk_reward_data,
        )
    )

    print(
        f"\nAligned dataset rows: "
        f"{len(dataset):,}"
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train_data, validation_data = (
        split_train_validation(
            dataset
        )
    )

    # --------------------------------------------------------
    # EMA SIGNALS
    # --------------------------------------------------------

    train_signal_data = (
        get_ema_signal_data(
            train_data
        )
    )

    validation_signal_data = (
        get_ema_signal_data(
            validation_data
        )
    )

    # --------------------------------------------------------
    # PRINT DATA SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "DATA SUMMARY"
    )
    print(
        "=" * 70
    )

    print(
        f"\nTraining period:"
    )

    if len(train_data) > 0:
        print(
            f"  Start: "
            f"{train_data['timestamp'].iloc[0]}"
        )
        print(
            f"  End:   "
            f"{train_data['timestamp'].iloc[-1]}"
        )

    print(
        f"  Total RR bars: "
        f"{len(train_data):,}"
    )

    print(
        f"  EMA crossover signals: "
        f"{len(train_signal_data):,}"
    )

    print(
        f"\nValidation period:"
    )

    if len(validation_data) > 0:
        print(
            f"  Start: "
            f"{validation_data['timestamp'].iloc[0]}"
        )
        print(
            f"  End:   "
            f"{validation_data['timestamp'].iloc[-1]}"
        )

    print(
        f"  Total RR bars: "
        f"{len(validation_data):,}"
    )

    print(
        f"  EMA crossover signals: "
        f"{len(validation_signal_data):,}"
    )

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "TRAINING TARGET"
    )
    print(
        "=" * 70
    )

    if len(train_signal_data) > 0:

        target_counts = (
            train_signal_data[
                "target"
            ]
            .value_counts()
            .sort_index()
        )

        losing = int(
            target_counts.get(
                0,
                0,
            )
        )

        winning = int(
            target_counts.get(
                1,
                0,
            )
        )

        total = losing + winning

        print(
            f"\nNot profitable: "
            f"{losing:,}"
        )

        print(
            f"Profitable:     "
            f"{winning:,}"
        )

        print(
            f"Total signals:  "
            f"{total:,}"
        )

        if total > 0:
            print(
                f"Training win rate: "
                f"{winning / total * 100:.2f}%"
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_prepared_data(
        client,
        dataset,
        train_signal_data,
        validation_signal_data,
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "DATA PREPARATION COMPLETE"
    )

    print(
        "=" * 70
    )

    return {
        "dataset": dataset,
        "train_data": train_data,
        "validation_data": validation_data,
        "train_signal_data": train_signal_data,
        "validation_signal_data": validation_signal_data,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    main()
