import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PERCENT = 0.70

MAX_ESTIMATORS = 20

EARLY_STOPPING_ROUNDS = 5

LEARNING_RATE = 0.05

MAX_DEPTH = 6

SUBSAMPLE = 0.8

COLSAMPLE_BYTREE = 0.8

RANDOM_STATE = 42


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def chronological_split(
    data,
):
    """
    Split data chronologically.

    First 70%:
        training

    Last 30%:
        validation
    """

    split_index = int(
        len(data)
        * TRAIN_PERCENT
    )

    if split_index <= 0:
        raise ValueError(
            "Training dataset is empty."
        )

    if split_index >= len(data):
        raise ValueError(
            "Validation dataset is empty."
        )

    training_data = (
        data.iloc[
            :split_index
        ]
        .reset_index(
            drop=True
        )
    )

    validation_data = (
        data.iloc[
            split_index:
        ]
        .reset_index(
            drop=True
        )
    )

    return (
        training_data,
        validation_data,
    )


# ============================================================
# MODEL
# ============================================================

def create_model(
    scale_pos_weight=1.0,
):
    """
    Create the GPU XGBoost classifier.

    Maximum of 20 estimators.

    Early stopping may stop training before
    reaching 20 estimators.
    """

    model = xgb.XGBClassifier(
        objective="binary:logistic",

        n_estimators=MAX_ESTIMATORS,

        learning_rate=LEARNING_RATE,

        max_depth=MAX_DEPTH,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE_BYTREE,

        min_child_weight=1,

        gamma=0.0,

        reg_alpha=0.0,

        reg_lambda=1.0,

        scale_pos_weight=scale_pos_weight,

        eval_metric="logloss",

        tree_method="hist",

        device="cuda",

        random_state=RANDOM_STATE,

        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS
        ),
    )

    return model


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(
    model,
    feature_names,
    feature_metadata=None,
):
    """
    Calculate feature usefulness.

    Every feature is included, even if its
    importance is zero.

    XGBoost provides several importance types:

        weight
        gain
        cover

    Gain is the primary importance measure.
    """

    booster = model.get_booster()

    weight_importance = (
        booster.get_score(
            importance_type="weight"
        )
    )

    gain_importance = (
        booster.get_score(
            importance_type="gain"
        )
    )

    cover_importance = (
        booster.get_score(
            importance_type="cover"
        )
    )

    rows = []

    for feature_name in feature_names:

        weight = (
            weight_importance
            .get(
                feature_name,
                0.0,
            )
        )

        gain = (
            gain_importance
            .get(
                feature_name,
                0.0,
            )
        )

        cover = (
            cover_importance
            .get(
                feature_name,
                0.0,
            )
        )

        rows.append(
            {
                "feature_name": feature_name,
                "weight": weight,
                "gain": gain,
                "cover": cover,
            }
        )

    result = pd.DataFrame(
        rows
    )

    # ========================================================
    # NORMALIZED GAIN
    # ========================================================

    total_gain = (
        result["gain"]
        .sum()
    )

    if total_gain > 0:

        result["gain_percent"] = (
            result["gain"]
            / total_gain
            * 100.0
        )

    else:

        result["gain_percent"] = 0.0

    # ========================================================
    # ADD FEATURE METADATA
    # ========================================================

    if feature_metadata is not None:

        result = result.merge(
            feature_metadata,
            on="feature_name",
            how="left",
        )

    result = (
        result
        .sort_values(
            "gain",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    result["importance_rank"] = (
        np.arange(
            1,
            len(result) + 1,
        )
    )

    return result


# ============================================================
# TRAIN MODEL
# ============================================================

def train_xgboost_model(
    data,
    feature_names,
    feature_metadata=None,
):
    """
    Train one XGBoost binary classifier.

    Target:

        1 = profitable
        0 = not profitable
    """

    # ========================================================
    # SPLIT
    # ========================================================

    training_data, validation_data = (
        chronological_split(
            data
        )
    )

    # ========================================================
    # X / Y
    # ========================================================

    X_train = (
        training_data[
            feature_names
        ]
    )

    y_train = (
        training_data[
            "target"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    X_validation = (
        validation_data[
            feature_names
        ]
    )

    y_validation = (
        validation_data[
            "target"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    # ========================================================
    # CLASS BALANCE
    # ========================================================

    positive_count = np.sum(
        y_train == 1
    )

    negative_count = np.sum(
        y_train == 0
    )

    if positive_count > 0:

        scale_pos_weight = (
            negative_count
            / positive_count
        )

    else:

        scale_pos_weight = 1.0

    # ========================================================
    # MODEL
    # ========================================================

    model = create_model(
        scale_pos_weight=(
            scale_pos_weight
        )
    )

    # ========================================================
    # TRAIN
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

    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.int32
    )

    # ========================================================
    # METRICS
    # ========================================================

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

    if (
        len(
            np.unique(
                y_validation
            )
        )
        > 1
    ):

        auc = roc_auc_score(
            y_validation,
            probabilities,
        )

    else:

        auc = np.nan

    # ========================================================
    # BEST ITERATION
    # ========================================================

    best_iteration = getattr(
        model,
        "best_iteration",
        MAX_ESTIMATORS - 1,
    )

    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    evaluation_results = (
        model.evals_result()
    )

    validation_logloss = (
        evaluation_results[
            "validation_0"
        ][
            "logloss"
        ]
    )

    training_metrics = pd.DataFrame(
        {
            "iteration": np.arange(
                len(
                    validation_logloss
                )
            ),

            "validation_logloss":
                validation_logloss,
        }
    )

    # ========================================================
    # SUMMARY ROW
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "training_rows": len(
                    training_data
                ),

                "validation_rows": len(
                    validation_data
                ),

                "training_positive":
                    int(
                        positive_count
                    ),

                "training_negative":
                    int(
                        negative_count
                    ),

                "validation_positive":
                    int(
                        np.sum(
                            y_validation
                            == 1
                        )
                    ),

                "validation_negative":
                    int(
                        np.sum(
                            y_validation
                            == 0
                        )
                    ),

                "accuracy":
                    accuracy,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "roc_auc":
                    auc,

                "best_iteration":
                    int(
                        best_iteration
                    ),

                "estimators_requested":
                    MAX_ESTIMATORS,

                "early_stopping_rounds":
                    EARLY_STOPPING_ROUNDS,

                "scale_pos_weight":
                    scale_pos_weight,
            }
        ]
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = (
        calculate_feature_importance(
            model=model,
            feature_names=feature_names,
            feature_metadata=feature_metadata,
        )
    )

    return (
        model,
        feature_importance,
        training_metrics,
        summary,
    )