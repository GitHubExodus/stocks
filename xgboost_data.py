import numpy as np
import pandas as pd


# ============================================================
# BIN / SPLIT COUNT CONFIGURATION
# ============================================================

def get_split_count(
    input_name,
):
    """
    Return the number of XGBoost histogram bins
    associated with an input type.

    These values are retained from the previous
    state-space configuration.

    XGBoost itself does not require us to manually
    create these bins. They are used only as metadata
    describing the intended resolution of each feature.
    """

    # --------------------------------------------------------
    # Dollar volume
    # --------------------------------------------------------

    if "dollar_volume" in input_name:
        return 40

    # --------------------------------------------------------
    # Minutes
    # --------------------------------------------------------

    if (
        input_name == "current_time"
        or "minutes_away" in input_name
    ):
        return 30

    # --------------------------------------------------------
    # Percentage values
    # --------------------------------------------------------

    if (
        "ema_distance" in input_name
        or "dema_distance" in input_name
        or "vwap_distance" in input_name
        or "roc" in input_name
        or "return_standard_deviation" in input_name
        or "normalized_atr" in input_name
        or "pivot_high_distance" in input_name
        or "pivot_low_distance" in input_name
    ):
        return 15

    # --------------------------------------------------------
    # Everything else
    # --------------------------------------------------------

    return 10


# ============================================================
# TIMESTAMP MATCHING
# ============================================================

def match_rr_to_input_rows(
    rr_timestamps,
    input_timestamps,
):
    """
    Match each RR entry timestamp to the latest input
    timestamp inside the corresponding 5-minute candle.

    RR timestamps represent the 5-minute candle close.

    Example:

        RR:
            13:00

        Input:
            12:55
            12:57
            12:59
            13:00

        Match:
            13:00

    If 13:00 does not exist:

        Match:
            12:59
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
        input_timestamps.asi8
    )

    rr_ns = (
        rr_timestamps.asi8
    )

    five_minutes = (
        pd.Timedelta(
            minutes=5
        ).value
    )

    for rr_index, rr_end in enumerate(
        rr_ns
    ):

        rr_start = (
            rr_end
            - five_minutes
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
# INPUT PREPARATION
# ============================================================

def prepare_input_data(
    input_data,
):
    """
    Clean and prepare the Input Stage dataframe.
    """

    data = (
        input_data
        .copy()
    )

    data["timestamp"] = (
        pd.to_datetime(
            data["timestamp"],
            utc=True,
        )
        .dt.tz_convert(
            "America/New_York"
        )
    )

    data = (
        data
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

    return data


# ============================================================
# RR PREPARATION
# ============================================================

def prepare_rr_data(
    risk_reward_data,
):
    """
    Clean and prepare RR data.
    """

    data = (
        risk_reward_data
        .copy()
    )

    data["start_timestamp"] = (
        pd.to_datetime(
            data["start_timestamp"],
            utc=True,
        )
        .dt.tz_convert(
            "America/New_York"
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
# FEATURE DISCOVERY
# ============================================================

def get_feature_names(
    input_data,
):
    """
    Determine which Input Stage columns are usable
    as XGBoost features.

    Timestamp is excluded.

    Raw OHLC columns are excluded.

    Symbol is excluded.

    Everything else numeric is considered a feature.
    """

    excluded_columns = {
        "timestamp",
        "Symbol",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vol",
        "trade_count",
        "vwap",
    }

    feature_names = []

    for column in input_data.columns:

        if column in excluded_columns:
            continue

        if pd.api.types.is_numeric_dtype(
            input_data[column]
        ):

            feature_names.append(
                column
            )

    return feature_names


# ============================================================
# BUILD XGBOOST DATA
# ============================================================

def build_xgboost_dataset(
    input_data,
    risk_reward_data,
):
    """
    Pair Input Stage data with RR data and create
    the binary XGBoost target.

    Target:

        trade_return_percent > 0
            -> 1

        trade_return_percent <= 0
            -> 0

    The exact exit price is NOT required.

    Output contains:

        timestamp
        feature columns
        trade_return_percent
        target
    """

    input_data = prepare_input_data(
        input_data
    )

    risk_reward_data = prepare_rr_data(
        risk_reward_data
    )

    feature_names = get_feature_names(
        input_data
    )

    # ========================================================
    # MATCH RR TO INPUT
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
    # VALID MATCHES
    # ========================================================

    valid_match = (
        input_row_indices >= 0
    )

    rr_data = (
        risk_reward_data.loc[
            valid_match
        ]
        .reset_index(
            drop=True
        )
    )

    matched_input_indices = (
        input_row_indices[
            valid_match
        ]
    )

    input_rows = (
        input_data.iloc[
            matched_input_indices
        ]
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # BUILD OUTPUT
    # ========================================================

    result = input_rows[
        [
            "timestamp"
        ]
        + feature_names
    ].copy()

    result["trade_return_percent"] = (
        rr_data[
            "trade_return_percent"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # ========================================================
    # TARGET
    # ========================================================

    result["target"] = (
        result[
            "trade_return_percent"
        ]
        > 0.0
    ).astype(
        np.int8
    )

    # ========================================================
    # REMOVE INVALID FEATURE ROWS
    # ========================================================

    feature_values = (
        result[
            feature_names
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    valid_features = np.all(
        np.isfinite(
            feature_values
        ),
        axis=1,
    )

    valid_returns = np.isfinite(
        result[
            "trade_return_percent"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    valid = (
        valid_features
        & valid_returns
    )

    result = (
        result.loc[
            valid
        ]
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # FINAL SORT
    # ========================================================

    result = (
        result
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        result,
        feature_names,
    )


# ============================================================
# FEATURE METADATA
# ============================================================

def build_feature_metadata(
    feature_names,
):
    """
    Create metadata for every feature.

    The split_count is retained as a useful description
    of the intended feature resolution.
    """

    rows = []

    for feature_name in feature_names:

        rows.append(
            {
                "feature_name": feature_name,
                "split_count": (
                    get_split_count(
                        feature_name
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )