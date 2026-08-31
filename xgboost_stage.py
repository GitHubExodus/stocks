import io
import traceback

import pandas as pd

from cloud_access import (
    download_input_data,
    download_risk_reward_data,

    save_xgboost_data,
    save_xgboost_model,
    save_xgboost_feature_importance,
    save_xgboost_training_metrics,

    log,
)

from xgboost_data import (
    build_xgboost_dataset,
    build_feature_metadata,
)

from xgboost_model import (
    train_xgboost_model,
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


# ============================================================
# MODEL SERIALIZATION
# ============================================================

def model_to_bytes(
    model,
):
    """
    Serialize XGBoost model to JSON bytes.
    """

    buffer = io.BytesIO()

    model.save_model(
        "/tmp/xgboost_model.json"
    )

    with open(
        "/tmp/xgboost_model.json",
        "rb",
    ) as file:

        return file.read()


# ============================================================
# TRAIN ONE STRATEGY
# ============================================================

def train_strategy(
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
    input_data,
):
    """
    Prepare and train one:

        trade_type
        stop loss
        RR

    combination.
    """

    log(
        "============================================================"
    )

    log(
        f"XGBOOST STRATEGY START | "
        f"symbol={symbol} | "
        f"trade_type={trade_type} | "
        f"SL={stop_loss_percentage} | "
        f"RR={risk_reward_ratio}"
    )

    # ========================================================
    # DOWNLOAD RR
    # ========================================================

    log(
        f"Downloading RR data | "
        f"symbol={symbol} | "
        f"trade_type={trade_type} | "
        f"SL={stop_loss_percentage} | "
        f"RR={risk_reward_ratio}"
    )

    risk_reward_data = (
        download_risk_reward_data(
            symbol=symbol,
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

    # ========================================================
    # BUILD XGBOOST DATASET
    # ========================================================

    log(
        "Preparing XGBoost dataset"
    )

    (
        xgboost_data,
        feature_names,
    ) = build_xgboost_dataset(
        input_data=input_data,
        risk_reward_data=risk_reward_data,
    )

    log(
        f"XGBoost dataset prepared | "
        f"rows={len(xgboost_data):,} | "
        f"features={len(feature_names)}"
    )

    # ========================================================
    # CLASS COUNTS
    # ========================================================

    positive_count = int(
        (
            xgboost_data[
                "target"
            ]
            == 1
        )
        .sum()
    )

    negative_count = int(
        (
            xgboost_data[
                "target"
            ]
            == 0
        )
        .sum()
    )

    log(
        f"Target distribution | "
        f"YES={positive_count:,} | "
        f"NO={negative_count:,}"
    )

    # ========================================================
    # SAVE PREPARED DATA
    # ========================================================

    log(
        "Saving prepared XGBoost data"
    )

    save_xgboost_data(
        symbol=symbol,
        trade_type=trade_type,
        stop_loss_percentage=(
            stop_loss_percentage
        ),
        risk_reward_ratio=(
            risk_reward_ratio
        ),
        xgboost_data=xgboost_data,
    )

    # ========================================================
    # FEATURE METADATA
    # ========================================================

    feature_metadata = (
        build_feature_metadata(
            feature_names
        )
    )

    # ========================================================
    # TRAIN
    # ========================================================

    log(
        "Starting GPU XGBoost training"
    )

    (
        model,
        feature_importance,
        training_metrics,
        summary,
    ) = train_xgboost_model(
        data=xgboost_data,
        feature_names=feature_names,
        feature_metadata=(
            feature_metadata
        ),
    )

    # ========================================================
    # LOG RESULTS
    # ========================================================

    metrics = (
        summary.iloc[0]
    )

    log(
        f"XGBoost training complete | "
        f"best_iteration="
        f"{metrics['best_iteration']} | "
        f"accuracy="
        f"{metrics['accuracy']:.4f} | "
        f"precision="
        f"{metrics['precision']:.4f} | "
        f"recall="
        f"{metrics['recall']:.4f} | "
        f"F1="
        f"{metrics['f1']:.4f} | "
        f"AUC="
        f"{metrics['roc_auc']:.4f}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    log(
        "Saving XGBoost model"
    )

    model_bytes = model_to_bytes(
        model
    )

    save_xgboost_model(
        symbol=symbol,
        trade_type=trade_type,
        stop_loss_percentage=(
            stop_loss_percentage
        ),
        risk_reward_ratio=(
            risk_reward_ratio
        ),
        model_bytes=model_bytes,
    )

    # ========================================================
    # SAVE FEATURE IMPORTANCE
    # ========================================================

    log(
        "Saving feature importance"
    )

    save_xgboost_feature_importance(
        symbol=symbol,
        trade_type=trade_type,
        stop_loss_percentage=(
            stop_loss_percentage
        ),
        risk_reward_ratio=(
            risk_reward_ratio
        ),
        feature_importance=(
            feature_importance
        ),
    )

    # ========================================================
    # SAVE TRAINING METRICS
    # ========================================================

    training_metrics = pd.concat(
        [
            training_metrics,
            summary.assign(
                iteration=-1
            ),
        ],
        ignore_index=True,
    )

    save_xgboost_training_metrics(
        symbol=symbol,
        trade_type=trade_type,
        stop_loss_percentage=(
            stop_loss_percentage
        ),
        risk_reward_ratio=(
            risk_reward_ratio
        ),
        training_metrics=(
            training_metrics
        ),
    )

    # ========================================================
    # DISPLAY TOP FEATURES
    # ========================================================

    log(
        "Top XGBoost features:"
    )

    top_features = (
        feature_importance
        .head(20)
    )

    for _, row in (
        top_features.iterrows()
    ):

        log(
            f"    {row['feature_name']} | "
            f"gain={row['gain_percent']:.4f}% | "
            f"split_count={row['split_count']}"
        )

    log(
        f"XGBOOST STRATEGY COMPLETE | "
        f"symbol={symbol} | "
        f"trade_type={trade_type} | "
        f"SL={stop_loss_percentage} | "
        f"RR={risk_reward_ratio}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log(
        "============================================================"
    )

    log(
        "XGBOOST STAGE STARTED"
    )

    log(
        "============================================================"
    )

    log(
        f"Symbol={SYMBOL}"
    )

    # ========================================================
    # DOWNLOAD INPUT ONCE
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

                try:

                    train_strategy(
                        symbol=SYMBOL,
                        trade_type=(
                            trade_type
                        ),
                        stop_loss_percentage=(
                            stop_loss_percentage
                        ),
                        risk_reward_ratio=(
                            risk_reward_ratio
                        ),
                        input_data=input_data,
                    )

                except Exception as error:

                    log(
                        f"XGBOOST STRATEGY FAILED | "
                        f"symbol={SYMBOL} | "
                        f"trade_type={trade_type} | "
                        f"SL={stop_loss_percentage} | "
                        f"RR={risk_reward_ratio} | "
                        f"error={error}"
                    )

                    log(
                        traceback.format_exc()
                    )

    log(
        "============================================================"
    )

    log(
        "XGBOOST STAGE COMPLETE"
    )

    log(
        "============================================================"
    )


if __name__ == "__main__":
    main()