import os
import io
import json
import traceback

import numpy as np
import pandas as pd
import xgboost as xgb

from cloud_access import (
    download_input_data,
    download_risk_reward_data,
    log,
    log_error,
)


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL = "DELL"

TRADE_TYPES = [
    "long",
    "short",
]

STOP_LOSS_PERCENTAGES = [
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
]

RISK_REWARD_RATIOS = [
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
]

# XGBoost probability required to enter a trade.
#
# Example:
#
#     0.70
#
# means:
#
#     model probability >= 70%
#         -> trade
#
#     model probability < 70%
#         -> no trade
#
PREDICTION_THRESHOLD = 0.70


# ============================================================
# R2 MODEL PATH
# ============================================================

XGBOOST_MODEL_PATH = "xgboost_models"


# ============================================================
# R2 EVALUATION PATH
# ============================================================

XGBOOST_EVALUATION_PATH = (
    "evaluate/xgboost"
)


# ============================================================
# IMPORT SHARED XGBOOST DATA FUNCTIONS
# ============================================================

from xgboost_data import (
    prepare_input_data,
    prepare_rr_data,
    get_feature_names,
    build_xgboost_dataset,
)


# ============================================================
# STRATEGY NUMBER FORMATTER
# ============================================================

def format_strategy_number(
    value,
):
    """
    Format strategy numbers exactly like
    the rest of the project.

    Examples:

        1       -> 1.0
        1.5     -> 1.5
        5       -> 5.0
    """

    return f"{float(value):.1f}"


# ============================================================
# MODEL PATH
# ============================================================

def get_model_key(
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
):
    """
    Return the R2 key for one trained XGBoost model.

    Example:

        xgboost_models/
            DELL/
                long/
                    sl_1.0_rr_2.0/
                        model.json
    """

    sl_string = format_strategy_number(
        stop_loss_percentage
    )

    rr_string = format_strategy_number(
        risk_reward_ratio
    )

    return (
        f"{XGBOOST_MODEL_PATH}/"
        f"{symbol}/"
        f"{trade_type}/"
        f"sl_{sl_string}_"
        f"rr_{rr_string}/"
        f"model.json"
    )


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def download_xgboost_model(
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
):
    """
    Download a trained XGBoost model from R2.

    Returns:

        xgb.Booster
    """

    key = get_model_key(
        symbol=symbol,
        trade_type=trade_type,
        stop_loss_percentage=(
            stop_loss_percentage
        ),
        risk_reward_ratio=(
            risk_reward_ratio
        ),
    )

    log(
        f"Downloading XGBoost model | "
        f"key={key}"
    )

    # --------------------------------------------------------
    # Reuse the S3 client from cloud_access.
    #
    # cloud_access.py already owns the R2 configuration.
    # --------------------------------------------------------

    import cloud_access

    response = cloud_access.s3.get_object(
        Bucket=cloud_access.R2_BUCKET_NAME,
        Key=key,
    )

    model_bytes = (
        response["Body"]
        .read()
    )

    model = xgb.Booster()

    model.load_model(
        bytearray(model_bytes)
    )

    log(
        f"XGBoost model downloaded | "
        f"symbol={symbol} | "
        f"trade_type={trade_type} | "
        f"SL={stop_loss_percentage} | "
        f"RR={risk_reward_ratio}"
    )

    return model


# ============================================================
# SAVE EQUITY CURVE
# ============================================================

def save_xgboost_equity_curve(
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
    equity_curve,
):
    """
    Save an XGBoost evaluation equity curve.

    Example:

        evaluate/xgboost/
            DELL/
                long/
                    sl_1.0_rr_2.0/
                        equity_curve.parquet
    """

    import cloud_access

    sl_string = format_strategy_number(
        stop_loss_percentage
    )

    rr_string = format_strategy_number(
        risk_reward_ratio
    )

    strategy_path = (
        f"{XGBOOST_EVALUATION_PATH}/"
        f"{symbol}/"
        f"{trade_type}/"
        f"sl_{sl_string}_"
        f"rr_{rr_string}"
    )

    cloud_access.ensure_r2_folder(
        XGBOOST_EVALUATION_PATH
    )

    cloud_access.ensure_r2_folder(
        f"{XGBOOST_EVALUATION_PATH}/{symbol}"
    )

    cloud_access.ensure_r2_folder(
        f"{XGBOOST_EVALUATION_PATH}/{symbol}/"
        f"{trade_type}"
    )

    cloud_access.ensure_r2_folder(
        strategy_path
    )

    key = (
        f"{strategy_path}/"
        f"equity_curve.parquet"
    )

    buffer = io.BytesIO()

    equity_curve.to_parquet(
        buffer,
        index=False,
    )

    buffer.seek(0)

    cloud_access.s3.put_object(
        Bucket=cloud_access.R2_BUCKET_NAME,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    log(
        f"XGBoost equity curve saved | "
        f"key={key}"
    )


# ============================================================
# BUILD EVALUATION EQUITY CURVE
# ============================================================

def evaluate_xgboost(
    input_data,
    risk_reward_data,
    model,
    prediction_threshold,
):
    """
    Run the trained XGBoost model over every
    matched RR/input bar.

    Process:

        Input bar
             ↓
        XGBoost prediction
             ↓
        probability >= threshold?
             ↓
        YES -> take RR trade
        NO  -> skip RR trade

    The actual RR trade_return_percent is then
    used for the equity curve.

    No price simulation is performed here.
    """

    # ========================================================
    # BUILD THE EXACT SAME DATASET USED FOR TRAINING
    # ========================================================

    dataset, feature_names = (
        build_xgboost_dataset(
            input_data=input_data,
            risk_reward_data=risk_reward_data,
        )
    )

    if len(dataset) == 0:

        return (
            pd.DataFrame(
                columns=[
                    "timestamp",
                    "trade_return_percent",
                    "prediction_probability",
                    "prediction",
                    "equity",
                ]
            ),
            feature_names,
        )

    # ========================================================
    # FEATURE MATRIX
    # ========================================================

    X = (
        dataset[
            feature_names
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    # ========================================================
    # XGBOOST PREDICTION
    # ========================================================

    dmatrix = xgb.DMatrix(
        X,
        feature_names=feature_names,
    )

    prediction_probability = (
        model.predict(
            dmatrix
        )
    )

    prediction_probability = (
        np.asarray(
            prediction_probability,
            dtype=np.float64,
        )
    )

    # ========================================================
    # TRADE / NO TRADE
    # ========================================================

    take_trade = (
        prediction_probability
        >= prediction_threshold
    )

    # ========================================================
    # ACTUAL RR RETURNS
    # ========================================================

    trade_returns = (
        dataset[
            "trade_return_percent"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    timestamps = (
        dataset[
            "timestamp"
        ]
        .to_numpy()
    )

    # ========================================================
    # BUILD EQUITY
    # ========================================================

    current_equity = 1.0

    output_timestamps = []
    output_returns = []
    output_probabilities = []
    output_predictions = []
    output_equity = []

    for index in range(
        len(dataset)
    ):

        # ----------------------------------------------------
        # Skip bars where the model says not to trade.
        # ----------------------------------------------------

        if not take_trade[index]:
            continue

        trade_return = (
            trade_returns[index]
        )

        if not np.isfinite(
            trade_return
        ):
            continue

        # ----------------------------------------------------
        # Apply actual RR result.
        #
        # Example:
        #
        # +2.0%
        #
        # equity:
        #
        # 1.00 -> 1.02
        #
        # -1.0%
        #
        # 1.02 -> 1.0098
        # ----------------------------------------------------

        current_equity *= (
            1.0
            + trade_return / 100.0
        )

        output_timestamps.append(
            timestamps[index]
        )

        output_returns.append(
            trade_return
        )

        output_probabilities.append(
            prediction_probability[index]
        )

        output_predictions.append(
            1
        )

        output_equity.append(
            current_equity
        )

    # ========================================================
    # CREATE RESULT
    # ========================================================

    equity_curve = pd.DataFrame(
        {
            "timestamp":
                output_timestamps,

            "trade_return_percent":
                output_returns,

            "prediction_probability":
                output_probabilities,

            "prediction":
                output_predictions,

            "equity":
                output_equity,
        }
    )

    return (
        equity_curve,
        feature_names,
    )


# ============================================================
# MAIN STRATEGY LOOP
# ============================================================

def main():

    log(
        "=================================================="
    )

    log(
        "XGBOOST EVALUATION STAGE STARTED"
    )

    log(
        "=================================================="
    )

    log(
        f"Symbol={SYMBOL}"
    )

    log(
        f"Prediction threshold="
        f"{PREDICTION_THRESHOLD}"
    )

    # ========================================================
    # DOWNLOAD INPUT DATA ONCE
    # ========================================================

    log(
        f"Downloading input data | "
        f"symbol={SYMBOL}"
    )

    input_data = (
        download_input_data(
            SYMBOL
        )
    )

    log(
        f"Input data downloaded | "
        f"rows={len(input_data):,} | "
        f"columns={len(input_data.columns):,}"
    )

    # ========================================================
    # EVERY STRATEGY
    # ========================================================

    for trade_type in TRADE_TYPES:

        for stop_loss_percentage in (
            STOP_LOSS_PERCENTAGES
        ):

            for risk_reward_ratio in (
                RISK_REWARD_RATIOS
            ):

                log(
                    "--------------------------------------------------"
                )

                log(
                    f"EVALUATION START | "
                    f"symbol={SYMBOL} | "
                    f"trade_type={trade_type} | "
                    f"SL={stop_loss_percentage} | "
                    f"RR={risk_reward_ratio}"
                )

                try:

                    # =================================================
                    # DOWNLOAD MODEL
                    # =================================================

                    model = (
                        download_xgboost_model(
                            symbol=SYMBOL,
                            trade_type=trade_type,
                            stop_loss_percentage=(
                                stop_loss_percentage
                            ),
                            risk_reward_ratio=(
                                risk_reward_ratio
                            ),
                        )
                    )

                    # =================================================
                    # DOWNLOAD RR DATA
                    # =================================================

                    log(
                        f"Downloading RR data | "
                        f"symbol={SYMBOL} | "
                        f"trade_type={trade_type} | "
                        f"SL={stop_loss_percentage} | "
                        f"RR={risk_reward_ratio}"
                    )

                    risk_reward_data = (
                        download_risk_reward_data(
                            symbol=SYMBOL,
                            trade_type=trade_type,
                            stop_loss_percentage=(
                                stop_loss_percentage
                            ),
                            risk_reward_ratio=(
                                risk_reward_ratio
                            ),
                        )
                    )

                    log(
                        f"RR data downloaded | "
                        f"rows={len(risk_reward_data):,}"
                    )

                    # =================================================
                    # EVALUATE
                    # =================================================

                    (
                        equity_curve,
                        feature_names,
                    ) = evaluate_xgboost(
                        input_data=input_data,
                        risk_reward_data=(
                            risk_reward_data
                        ),
                        model=model,
                        prediction_threshold=(
                            PREDICTION_THRESHOLD
                        ),
                    )

                    # =================================================
                    # SAVE
                    # =================================================

                    save_xgboost_equity_curve(
                        symbol=SYMBOL,
                        trade_type=trade_type,
                        stop_loss_percentage=(
                            stop_loss_percentage
                        ),
                        risk_reward_ratio=(
                            risk_reward_ratio
                        ),
                        equity_curve=(
                            equity_curve
                        ),
                    )

                    # =================================================
                    # STATISTICS
                    # =================================================

                    total_trades = (
                        len(equity_curve)
                    )

                    if total_trades > 0:

                        final_equity = (
                            equity_curve[
                                "equity"
                            ]
                            .iloc[-1]
                        )

                        total_return = (
                            (
                                final_equity
                                - 1.0
                            )
                            * 100.0
                        )

                    else:

                        final_equity = 1.0
                        total_return = 0.0

                    log(
                        f"EVALUATION COMPLETE | "
                        f"symbol={SYMBOL} | "
                        f"trade_type={trade_type} | "
                        f"SL={stop_loss_percentage} | "
                        f"RR={risk_reward_ratio} | "
                        f"trades={total_trades:,} | "
                        f"final_equity={final_equity:.6f} | "
                        f"return={total_return:.2f}%"
                    )

                except Exception as error:

                    log_error(
                        stage=(
                            "XGBOOST_EVALUATION"
                        ),
                        symbol=SYMBOL,
                        error=error,
                        context=(
                            f"trade_type={trade_type}, "
                            f"SL={stop_loss_percentage}, "
                            f"RR={risk_reward_ratio}"
                        ),
                    )

                    # ------------------------------------------------
                    # Continue to the next strategy instead of
                    # terminating the entire evaluation.
                    # ------------------------------------------------

                    continue

    # ========================================================
    # COMPLETE
    # ========================================================

    log(
        "=================================================="
    )

    log(
        "XGBOOST EVALUATION STAGE COMPLETE"
    )

    log(
        "=================================================="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
