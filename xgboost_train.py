# ==========================================================
# CELL 1 - CONFIGURATION
# ==========================================================

import gc
import json
import os
import shutil
import time
import warnings

from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)

warnings.filterwarnings("ignore")


# ==========================================================
# PROJECT SETTINGS
# ==========================================================

RANDOM_STATE = 42

TRAIN_SPLIT = 0.60
VALID_SPLIT = 0.20
TEST_SPLIT = 0.20

REMOVE_PERCENT = 0.10
REMOVE_ONE_BELOW = 20

USE_GPU = True

EARLY_STOPPING_ROUNDS = 100

N_ESTIMATORS = 5000


# ==========================================================
# CLOUDFLARE R2
# ==========================================================

R2_ENDPOINT = "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"

R2_BUCKET = "stocks-data"

R2_ACCESS_KEY = "00e18b0c16ecb3395cd6f7c8e0eb3554"

R2_SECRET_KEY = "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"


# ==========================================================
# LOCAL FOLDERS
# ==========================================================

ROOT = Path.cwd()

DATA_DIR = ROOT / "data"

RAW_INPUT_DIR = DATA_DIR / "raw_input"
RAW_OUTPUT_DIR = DATA_DIR / "raw_output"

MERGED_DIR = DATA_DIR / "merged"

MODELS_DIR = ROOT / "models"

LEADERBOARD_PATH = ROOT / "leaderboard.csv"


# ==========================================================
# CREATE FOLDERS
# ==========================================================

for folder in [

    DATA_DIR,

    RAW_INPUT_DIR,
    RAW_OUTPUT_DIR,

    MERGED_DIR,

    MODELS_DIR,

]:

    folder.mkdir(parents=True, exist_ok=True)


# ==========================================================
# FIXED XGBOOST PARAMETERS
# ==========================================================

XGB_FIXED_PARAMS = {

    "tree_method": "hist",

    "device": "cuda" if USE_GPU else "cpu",

    "n_estimators": N_ESTIMATORS,

    "verbosity": 0,

    "random_state": RANDOM_STATE,

}


# ==========================================================
# HYPERPARAMETER PRESETS
# ==========================================================

HYPERPARAMETERS = [

    {
        "id": 1,
        "learning_rate": 0.005,
        "max_depth": 4,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1,
        "reg_alpha": 0,
    },

    {
        "id": 2,
        "learning_rate": 0.01,
        "max_depth": 6,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1,
        "reg_alpha": 0,
    },

    {
        "id": 3,
        "learning_rate": 0.02,
        "max_depth": 6,
        "min_child_weight": 3,
        "gamma": 0.3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 5,
        "reg_alpha": 0,
    },

    {
        "id": 4,
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 3,
        "gamma": 0.3,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "reg_lambda": 5,
        "reg_alpha": 0,
    },

    {
        "id": 5,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "gamma": 0.3,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
        "reg_alpha": 0,
    },

    {
        "id": 6,
        "learning_rate": 0.075,
        "max_depth": 8,
        "min_child_weight": 5,
        "gamma": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
        "reg_alpha": 1,
    },

    {
        "id": 7,
        "learning_rate": 0.10,
        "max_depth": 10,
        "min_child_weight": 5,
        "gamma": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 10,
        "reg_alpha": 1,
    },

    {
        "id": 8,
        "learning_rate": 0.15,
        "max_depth": 10,
        "min_child_weight": 10,
        "gamma": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "reg_lambda": 10,
        "reg_alpha": 1,
    },

    {
        "id": 9,
        "learning_rate": 0.20,
        "max_depth": 12,
        "min_child_weight": 10,
        "gamma": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "reg_lambda": 10,
        "reg_alpha": 2,
    },

    {
        "id": 10,
        "learning_rate": 0.30,
        "max_depth": 12,
        "min_child_weight": 10,
        "gamma": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 10,
        "reg_alpha": 2,
    },

]


# ==========================================================
# READY
# ==========================================================

print("=" * 60)
print("Configuration Loaded")
print("=" * 60)

print(f"GPU Enabled            : {USE_GPU}")
print(f"Hyperparameter Presets : {len(HYPERPARAMETERS)}")
print(f"Train Split            : {TRAIN_SPLIT:.0%}")
print(f"Validation Split       : {VALID_SPLIT:.0%}")
print(f"Test Split             : {TEST_SPLIT:.0%}")

print("=" * 60)

































# ==========================================================
# CELL 2 - CONNECT TO CLOUDFLARE R2
# ==========================================================

print("=" * 60)
print("Connecting to Cloudflare R2...")
print("=" * 60)

session = boto3.session.Session()

s3 = session.client(
    service_name="s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
)

try:

    s3.head_bucket(Bucket=R2_BUCKET)

    print(f"Connected to bucket: {R2_BUCKET}")

except Exception as e:

    print("Failed to connect to R2.")
    raise e


# ----------------------------------------------------------
# VERIFY REQUIRED PREFIXES EXIST
# ----------------------------------------------------------

def prefix_exists(prefix):

    response = s3.list_objects_v2(
        Bucket=R2_BUCKET,
        Prefix=prefix + "/",
        MaxKeys=1,
    )

    return "Contents" in response


print()

if prefix_exists("input"):

    print("Found: input/")

else:

    raise Exception("'input/' not found or contains no files.")


if prefix_exists("output"):

    print("Found: output/")

else:

    raise Exception("'output/' not found or contains no files.")


print()
print("Connection Successful.")




































# ==========================================================
# CELL 3 - DOWNLOAD ALL INPUT / OUTPUT FILES FROM R2
# ==========================================================

print("=" * 60)
print("Downloading Files From Cloudflare R2...")
print("=" * 60)


# ----------------------------------------------------------
# CREATE R2 FOLDER IF MISSING
# ----------------------------------------------------------

def ensure_r2_folder(folder_name):

    key = folder_name.rstrip("/") + "/"

    try:

        s3.head_object(
            Bucket=R2_BUCKET,
            Key=key,
        )

    except:

        s3.put_object(
            Bucket=R2_BUCKET,
            Key=key,
            Body=b"",
        )

        print(f"Created R2 folder: {folder_name}")


# These folders will eventually contain results
ensure_r2_folder("models")
ensure_r2_folder("leaderboards")
ensure_r2_folder("metadata")


# ----------------------------------------------------------
# DOWNLOAD FUNCTION
# ----------------------------------------------------------

def download_folder(r2_folder, local_folder, start=0, limit=None):

    paginator = s3.get_paginator("list_objects_v2")

    pages = paginator.paginate(
        Bucket=R2_BUCKET,
        Prefix=r2_folder + "/",
    )

    keys = []

    for page in pages:

        if "Contents" not in page:
            continue

        for obj in page["Contents"]:

            key = obj["Key"]

            if key.endswith("/"):
                continue

            keys.append(key)

    keys.sort()

    if limit is None:
        selected_keys = keys[start:]
    else:
        selected_keys = keys[start:start + limit]

    downloaded = 0

    for key in selected_keys:

        filename = os.path.basename(key)

        local_path = local_folder / filename

        if local_path.exists():
            continue

        print(f"Downloading {filename}")

        s3.download_file(
            R2_BUCKET,
            key,
            str(local_path),
        )

        downloaded += 1

    return downloaded


# ----------------------------------------------------------
# DOWNLOAD INPUTS
# ----------------------------------------------------------

START_STOCK = 2      # Start at the third stock (0-based indexing)
TEST_STOCKS = 1     # Download 10 stocks

input_count = download_folder(
    "input",
    RAW_INPUT_DIR,
    start=START_STOCK,
    limit=TEST_STOCKS,
)

print()
print(f"Downloaded {input_count} input files.")


# ----------------------------------------------------------
# DOWNLOAD OUTPUTS
# ----------------------------------------------------------

output_count = download_folder(
    "output",
    RAW_OUTPUT_DIR,
    start=START_STOCK,
    limit=TEST_STOCKS,
)

print()
print(f"Downloaded {output_count} output files.")

print()
print("Download Complete.")


































# ==========================================================
# CELL 4 - SPLIT EACH STOCK INTO TRAIN / VALIDATION / TEST
# ==========================================================

print("=" * 60)
print("Splitting Stock Files...")
print("=" * 60)

TRAIN_INPUT_PATH = MERGED_DIR / "train_input.npy"
VALID_INPUT_PATH = MERGED_DIR / "valid_input.npy"
TEST_INPUT_PATH  = MERGED_DIR / "test_input.npy"

TRAIN_OUTPUT_PATH = MERGED_DIR / "train_output.npy"
VALID_OUTPUT_PATH = MERGED_DIR / "valid_output.npy"
TEST_OUTPUT_PATH  = MERGED_DIR / "test_output.npy"


train_input_list = []
valid_input_list = []
test_input_list = []

train_output_list = []
valid_output_list = []
test_output_list = []


input_files = sorted(RAW_INPUT_DIR.glob("*.parquet"))

# ------------------------------------------------------
# TEMPORARY LIMIT FOR TESTING
# ------------------------------------------------------

# input_files = input_files[:1]

if len(input_files) == 0:
    raise Exception("No input parquet files found.")


for input_path in input_files:

    output_path = RAW_OUTPUT_DIR / input_path.name

    if not output_path.exists():
        print(f"Missing output file: {input_path.name}")
        continue

    print(f"Processing {input_path.name}")

    # ------------------------------------------------------
    # LOAD
    # ------------------------------------------------------

    input_df = pd.read_parquet(input_path)

    output_df = pd.read_parquet(output_path)

    # Ignore timestamp column
    output_df = output_df.iloc[:, 1:]

    if len(input_df) != len(output_df):
        raise Exception(f"Row mismatch: {input_path.name}")

    rows = len(input_df)

    train_end = int(rows * TRAIN_SPLIT)
    valid_end = train_end + int(rows * VALID_SPLIT)

    # ------------------------------------------------------
    # INPUT
    # ------------------------------------------------------

    train_input_list.append(
        input_df.iloc[:train_end].to_numpy(dtype=np.float32)
    )

    valid_input_list.append(
        input_df.iloc[train_end:valid_end].to_numpy(dtype=np.float32)
    )

    test_input_list.append(
        input_df.iloc[valid_end:].to_numpy(dtype=np.float32)
    )

    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------

    train_output_list.append(
        output_df.iloc[:train_end].to_numpy(dtype=np.int8)
    )

    valid_output_list.append(
        output_df.iloc[train_end:valid_end].to_numpy(dtype=np.int8)
    )

    test_output_list.append(
        output_df.iloc[valid_end:].to_numpy(dtype=np.int8)
    )

    del input_df
    del output_df

    gc.collect()


# ==========================================================
# MERGE ALL STOCKS
# ==========================================================

print()
print("Merging datasets...")

train_input = np.concatenate(train_input_list, axis=0)
valid_input = np.concatenate(valid_input_list, axis=0)
test_input = np.concatenate(test_input_list, axis=0)

train_output = np.concatenate(train_output_list, axis=0)
valid_output = np.concatenate(valid_output_list, axis=0)
test_output = np.concatenate(test_output_list, axis=0)

del train_input_list
del valid_input_list
del test_input_list

del train_output_list
del valid_output_list
del test_output_list

gc.collect()


# ==========================================================
# SAVE
# ==========================================================

np.save(TRAIN_INPUT_PATH, train_input)
np.save(VALID_INPUT_PATH, valid_input)
np.save(TEST_INPUT_PATH, test_input)

np.save(TRAIN_OUTPUT_PATH, train_output)
np.save(VALID_OUTPUT_PATH, valid_output)
np.save(TEST_OUTPUT_PATH, test_output)


FEATURE_NAMES = pd.read_parquet(input_files[0]).columns.tolist()

STRATEGY_NAMES = (
    pd.read_parquet(RAW_OUTPUT_DIR / input_files[0].name)
    .columns
    .tolist()[1:]
)

with open(MERGED_DIR / "feature_names.json", "w") as f:
    json.dump(FEATURE_NAMES, f)

with open(MERGED_DIR / "strategy_names.json", "w") as f:
    json.dump(STRATEGY_NAMES, f)


print()
print("=" * 60)
print("Finished")
print("=" * 60)

print(f"Training Rows   : {len(train_input):,}")
print(f"Validation Rows : {len(valid_input):,}")
print(f"Testing Rows    : {len(test_input):,}")

print(f"Features        : {len(FEATURE_NAMES)}")
print(f"Strategies      : {len(STRATEGY_NAMES)}")











































# ==========================================================
# CELL 5 - CREATE LEADERBOARD
# ==========================================================

print("=" * 60)
print("Creating Leaderboard...")
print("=" * 60)


leaderboard_columns = [

    "experiment_id",

    "strategy",

    "hyperparameter_id",

    "elimination_round",

    "feature_count",

    "accuracy",

    "precision",

    "recall",

    "f1",

    "log_loss",

    "best_iteration",

    "training_time_seconds",

    "model_folder",

]


if not LEADERBOARD_PATH.exists():

    leaderboard = pd.DataFrame(columns=leaderboard_columns)

    leaderboard.to_csv(
        LEADERBOARD_PATH,
        index=False,
    )

    print("Leaderboard Created")

else:

    print("Leaderboard Already Exists")


# ----------------------------------------------------------
# EXPERIMENT COUNTER
# ----------------------------------------------------------

EXPERIMENT_COUNTER = 1

if MODELS_DIR.exists():

    existing = sorted(MODELS_DIR.glob("experiment_*"))

    if len(existing) > 0:

        numbers = []

        for folder in existing:

            try:

                numbers.append(
                    int(folder.name.split("_")[1])
                )

            except:
                pass

        if len(numbers) > 0:

            EXPERIMENT_COUNTER = max(numbers) + 1


print(f"Next Experiment ID : {EXPERIMENT_COUNTER:06d}")

print()
print("Leaderboard Ready.")








































# ==========================================================
# CELL 6 - HELPER FUNCTIONS
# ==========================================================

print("=" * 60)
print("Loading Datasets...")
print("=" * 60)


# ----------------------------------------------------------
# LOAD DATASETS
# ----------------------------------------------------------

TRAIN_INPUT = np.load(
    MERGED_DIR / "train_input.npy",
    mmap_mode="r",
)

VALID_INPUT = np.load(
    MERGED_DIR / "valid_input.npy",
    mmap_mode="r",
)

TEST_INPUT = np.load(
    MERGED_DIR / "test_input.npy",
    mmap_mode="r",
)

TRAIN_OUTPUT = np.load(
    MERGED_DIR / "train_output.npy",
    mmap_mode="r",
)

VALID_OUTPUT = np.load(
    MERGED_DIR / "valid_output.npy",
    mmap_mode="r",
)

TEST_OUTPUT = np.load(
    MERGED_DIR / "test_output.npy",
    mmap_mode="r",
)


# ----------------------------------------------------------
# LOAD FEATURE / STRATEGY NAMES
# ----------------------------------------------------------

with open(MERGED_DIR / "feature_names.json", "r") as f:

    FEATURE_NAMES = json.load(f)

with open(MERGED_DIR / "strategy_names.json", "r") as f:

    STRATEGY_NAMES = json.load(f)


def encode_labels(train_y, valid_y):

    classes = np.sort(np.unique(train_y))

    mapping = {
        cls: i
        for i, cls in enumerate(classes)
    }

    train_encoded = np.array(
        [mapping[v] for v in train_y],
        dtype=np.int32,
    )

    valid_encoded = np.array(
        [mapping[v] for v in valid_y],
        dtype=np.int32,
    )

    return train_encoded, valid_encoded, mapping


def decode_labels(y):

    return np.vectorize(REVERSE_LABEL_MAP.get)(y).astype(np.int8)


# ----------------------------------------------------------
# SAVE JSON
# ----------------------------------------------------------

def save_json(path, data):

    with open(path, "w") as f:

        json.dump(
            data,
            f,
            indent=4,
        )


# ----------------------------------------------------------
# APPEND TO LEADERBOARD
# ----------------------------------------------------------

def append_leaderboard(row):

    pd.DataFrame([row]).to_csv(
        LEADERBOARD_PATH,
        mode="a",
        header=False,
        index=False,
    )


# ----------------------------------------------------------
# CREATE EXPERIMENT FOLDER
# ----------------------------------------------------------

def create_experiment_folder(experiment_id):

    folder = MODELS_DIR / f"experiment_{experiment_id:06d}"

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return folder


print(f"Training Rows   : {TRAIN_INPUT.shape[0]:,}")
print(f"Validation Rows : {VALID_INPUT.shape[0]:,}")
print(f"Testing Rows    : {TEST_INPUT.shape[0]:,}")

print(f"Features        : {len(FEATURE_NAMES)}")
print(f"Strategies      : {len(STRATEGY_NAMES)}")

print()
print("Helper Functions Ready.")


























# ==========================================================
# CELL 7 - TRAIN ONE MODEL
# ==========================================================

def train_model(

    strategy_index,
    feature_indices,
    hyperparameters,

):

    # ------------------------------------------------------
    # BUILD DATA
    # ------------------------------------------------------

    x_train = TRAIN_INPUT[:, feature_indices]
    x_valid = VALID_INPUT[:, feature_indices]

    y_train, y_valid, label_mapping = encode_labels(

        TRAIN_OUTPUT[:, strategy_index],

        VALID_OUTPUT[:, strategy_index],

    )

    num_classes = len(label_mapping)

    # ------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------

    params = XGB_FIXED_PARAMS.copy()

    params.update(hyperparameters)

    params.pop("id", None)

    if num_classes == 2:

        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        params.pop("num_class", None)

    else:

        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        params["num_class"] = num_classes

    # params.update(hyperparameters)

    print(params)
    print("Classes:", np.unique(y_train))

    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------
    
    model = xgb.XGBClassifier(
        **params,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    start_time = time.time()

    model.fit(

        x_train,
        y_train,

        eval_set=[
            (x_valid, y_valid)
        ],

        verbose=False,

    )

    training_time = time.time() - start_time

    # ------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------

    predictions = model.predict(
        x_valid
    )

    probabilities = model.predict_proba(
        x_valid
    )

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    accuracy = accuracy_score(
        y_valid,
        predictions,
    )

    precision = precision_score(
        y_valid,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_valid,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_valid,
        predictions,
        average="macro",
        zero_division=0,
    )

    loss = log_loss(
        y_valid,
        probabilities,
    )

    # ------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({

        "feature":

            np.array(FEATURE_NAMES)[feature_indices],

        "importance":

            importance,

    })

    feature_importance = feature_importance.sort_values(

        "importance",

        ascending=False,

        ignore_index=True,

    )

    return {

        "model": model,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "log_loss": loss,

        "training_time": training_time,

        "best_iteration": model.best_iteration,

        "feature_importance": feature_importance,

    }


print("Training Function Ready.")

































# ==========================================================
# CELL 8 - FEATURE ELIMINATION + MODEL SAVING
# ==========================================================

def recursive_feature_elimination(

    strategy_index,
    hyperparameters,

):

    global EXPERIMENT_COUNTER

    strategy_name = STRATEGY_NAMES[strategy_index]

    remaining_features = np.arange(len(FEATURE_NAMES))

    removed_features = []

    elimination_round = 1

    while True:

        print(
            f"{strategy_name} | "
            f"Preset {hyperparameters['id']} | "
            f"Round {elimination_round} | "
            f"Features {len(remaining_features)}"
        )

        # --------------------------------------------------
        # TRAIN
        # --------------------------------------------------

        results = train_model(

            strategy_index=strategy_index,

            feature_indices=remaining_features,

            hyperparameters=hyperparameters,

        )

        # --------------------------------------------------
        # CREATE EXPERIMENT
        # --------------------------------------------------

        experiment_id = EXPERIMENT_COUNTER

        EXPERIMENT_COUNTER += 1

        experiment_folder = create_experiment_folder(
            experiment_id
        )

        # --------------------------------------------------
        # SAVE MODEL
        # --------------------------------------------------

        results["model"].save_model(
            experiment_folder / "model.json"
        )

        # --------------------------------------------------
        # SAVE FEATURE IMPORTANCE
        # --------------------------------------------------

        results["feature_importance"].to_csv(

            experiment_folder / "feature_importance.csv",

            index=False,

        )

        # --------------------------------------------------
        # SAVE REMOVED FEATURES
        # --------------------------------------------------

        pd.DataFrame(removed_features).to_csv(

            experiment_folder / "removed_features.csv",

            index=False,

        )

        # --------------------------------------------------
        # SAVE METADATA
        # --------------------------------------------------

        metadata = {

            "experiment_id": experiment_id,

            "strategy": strategy_name,

            "hyperparameter_id": hyperparameters["id"],

            "hyperparameters": hyperparameters,

            "feature_count": len(remaining_features),

            "features": list(
                np.array(FEATURE_NAMES)[remaining_features]
            ),

            "accuracy": results["accuracy"],

            "precision": results["precision"],

            "recall": results["recall"],

            "f1": results["f1"],

            "log_loss": results["log_loss"],

            "best_iteration": int(
                results["best_iteration"]
            ),

            "training_time_seconds": results[
                "training_time"
            ],

        }

        save_json(

            experiment_folder / "metadata.json",

            metadata,

        )

        # --------------------------------------------------
        # LEADERBOARD
        # --------------------------------------------------

        append_leaderboard({

            "experiment_id": experiment_id,

            "strategy": strategy_name,

            "hyperparameter_id": hyperparameters["id"],

            "elimination_round": elimination_round,

            "feature_count": len(remaining_features),

            "accuracy": results["accuracy"],

            "precision": results["precision"],

            "recall": results["recall"],

            "f1": results["f1"],

            "log_loss": results["log_loss"],

            "best_iteration": results["best_iteration"],

            "training_time_seconds":
                results["training_time"],

            "model_folder":
                experiment_folder.name,

        })

        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        if len(remaining_features) == 1:

            print()

            break

        # --------------------------------------------------
        # REMOVE FEATURES
        # --------------------------------------------------

        importance = results["feature_importance"]

        if len(remaining_features) > REMOVE_ONE_BELOW:

            remove_count = max(
                1,
                int(len(remaining_features) * REMOVE_PERCENT),
            )

        else:

            remove_count = 1

        worst = importance.tail(remove_count)

        for _, row in worst.iterrows():

            removed_features.append({

                "round": elimination_round,

                "feature": row["feature"],

                "importance": float(row["importance"]),

            })

        keep = importance.iloc[:-remove_count]

        remaining_features = np.array([

            FEATURE_NAMES.index(feature)

            for feature in keep["feature"]

        ])

        elimination_round += 1


print("Recursive Feature Elimination Ready.")







































# ==========================================================
# CELL 9 - MAIN TRAINING LOOP
# ==========================================================

print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

overall_start = time.time()

total_models = 0

for strategy_index, strategy_name in enumerate(STRATEGY_NAMES):

    print()
    print("=" * 60)
    print(f"STRATEGY: {strategy_name}")
    print("=" * 60)

    strategy_start = time.time()

    for hyperparameters in HYPERPARAMETERS:

        print()
        print("-" * 60)
        print(f"Preset {hyperparameters['id']}")
        print("-" * 60)

        recursive_feature_elimination(

            strategy_index=strategy_index,

            hyperparameters=hyperparameters,

        )

        total_models += 1

        gc.collect()

    strategy_time = time.time() - strategy_start

    print()
    print(f"Finished Strategy: {strategy_name}")
    print(f"Time: {strategy_time / 60:.2f} minutes")

overall_time = time.time() - overall_start

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Strategies           : {len(STRATEGY_NAMES)}")
print(f"Hyperparameter Sets  : {len(HYPERPARAMETERS)}")
print(f"Experiment Folders   : {EXPERIMENT_COUNTER - 1}")
print(f"Recursive Runs       : {total_models}")
print(f"Total Time (Hours)   : {overall_time / 3600:.2f}")







# ==========================================================
# CELL 10 - UPLOAD ALL MODELS TO CLOUDFLARE R2
# ==========================================================

print("=" * 60)
print("Uploading Models To Cloudflare R2...")
print("=" * 60)


# ----------------------------------------------------------
# UPLOAD FILE
# ----------------------------------------------------------

def upload_file(local_path, r2_key):

    s3.upload_file(
        str(local_path),
        R2_BUCKET,
        r2_key,
    )


# ----------------------------------------------------------
# UPLOAD MODELS
# ----------------------------------------------------------

model_count = 0
file_count = 0

for experiment_folder in sorted(MODELS_DIR.glob("experiment_*")):

    print(f"Uploading {experiment_folder.name}")

    for file in experiment_folder.rglob("*"):

        if not file.is_file():
            continue

        relative_path = file.relative_to(MODELS_DIR)

        r2_key = f"models/{relative_path.as_posix()}"

        upload_file(
            file,
            r2_key,
        )

        file_count += 1

    model_count += 1


# ----------------------------------------------------------
# UPLOAD LEADERBOARD
# ----------------------------------------------------------

if LEADERBOARD_PATH.exists():

    upload_file(

        LEADERBOARD_PATH,

        "leaderboards/leaderboard.csv",

    )

    print("Uploaded leaderboard.csv")


print()
print("=" * 60)
print("UPLOAD COMPLETE")
print("=" * 60)

print(f"Experiment Folders : {model_count}")
print(f"Files Uploaded     : {file_count + 1}")
print("Done.")