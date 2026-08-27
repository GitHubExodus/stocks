import io
import boto3
import pandas as pd


# ============================================================
# R2 CONFIGURATION
# ============================================================

R2_ACCESS_KEY_ID = "00e18b0c16ecb3395cd6f7c8e0eb3554"
R2_SECRET_ACCESS_KEY = "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"
R2_ENDPOINT_URL = "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"

R2_BUCKET_NAME = "stocks-data"


# ============================================================
# R2 PATHS
# ============================================================

STOCK_SYMBOLS_PATH = "misc/symbols.txt"
COMPLETED_SYMBOLS_PATH = "misc/completed_symbols.txt"

RAW_STOCK_DATA_PATH = ""
INPUT_PATH = "input"
INPUT_STATISTICS_PATH = "input_statistics"
RISK_REWARD_PATH = "riskreward"


# ============================================================
# R2 CLIENT
# ============================================================

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
)


def ensure_r2_folder(folder_path):
    """
    Make sure an R2 folder exists.
    """

    folder_path = (
        folder_path.rstrip("/")
        + "/"
    )

    try:

        s3.head_object(
            Bucket=R2_BUCKET_NAME,
            Key=folder_path,
        )

    except s3.exceptions.ClientError as error:

        error_code = (
            error.response["Error"]["Code"]
        )

        if error_code in (
            "404",
            "NoSuchKey",
        ):

            s3.put_object(
                Bucket=R2_BUCKET_NAME,
                Key=folder_path,
            )

        else:
            raise


# ============================================================
# LOGGING
# ============================================================

import traceback
import uuid
from datetime import datetime, timezone


LOGS_PATH = "logs"

DEFAULT_INSTANCE_ID = (
    uuid.uuid4().hex[:8]
)

INSTANCE_ID = DEFAULT_INSTANCE_ID

INSTANCE_LOG_PATH = (
    f"{LOGS_PATH}/{INSTANCE_ID}"
)

PROGRAM_LOG_PATH = (
    f"{INSTANCE_LOG_PATH}/program.log"
)

ERROR_LOG_PATH = (
    f"{INSTANCE_LOG_PATH}/errors.log"
)


def configure_logging(instance_id=None):
    """
    Configure the ID used by the logging system.

    If instance_id is supplied, use that ID.

    If instance_id is not supplied, use the
    automatically generated ID.
    """

    global INSTANCE_ID
    global INSTANCE_LOG_PATH
    global PROGRAM_LOG_PATH
    global ERROR_LOG_PATH

    if instance_id is None:
        instance_id = DEFAULT_INSTANCE_ID

    INSTANCE_ID = str(
        instance_id
    )

    INSTANCE_LOG_PATH = (
        f"{LOGS_PATH}/{INSTANCE_ID}"
    )

    PROGRAM_LOG_PATH = (
        f"{INSTANCE_LOG_PATH}/program.log"
    )

    ERROR_LOG_PATH = (
        f"{INSTANCE_LOG_PATH}/errors.log"
    )


def _append_log(
    log_path,
    text,
):
    """
    Append a timestamped message to an R2 log.

    Creates the log automatically if it
    does not already exist.
    """

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    log_entry = (
        f"[{timestamp}] "
        f"INSTANCE={INSTANCE_ID} | "
        f"{text}"
    )

    try:

        response = s3.get_object(
            Bucket=R2_BUCKET_NAME,
            Key=log_path,
        )

        existing_log = (
            response["Body"]
            .read()
            .decode("utf-8")
        )

    except s3.exceptions.ClientError as error:

        error_code = (
            error.response["Error"]["Code"]
        )

        if error_code in (
            "404",
            "NoSuchKey",
        ):

            existing_log = ""

        else:
            raise

    if existing_log:

        updated_log = (
            existing_log
            + "\n"
            + log_entry
        )

    else:

        updated_log = log_entry

    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=log_path,
        Body=updated_log.encode("utf-8"),
        ContentType="text/plain",
    )


def log(text):
    """
    Write a normal program message.
    """

    print(text)

    _append_log(
        PROGRAM_LOG_PATH,
        text,
    )


def log_warning(text):
    """
    Write a warning message.
    """

    print(
        f"WARNING: {text}"
    )

    _append_log(
        PROGRAM_LOG_PATH,
        f"WARNING | {text}",
    )


def log_error(
    stage,
    symbol,
    error,
    context=None,
):
    """
    Write an error with the complete traceback.
    """

    traceback_text = (
        traceback.format_exc()
    )

    message = (
        f"STAGE={stage} | "
        f"SYMBOL={symbol} | "
        f"ERROR={type(error).__name__} | "
        f"MESSAGE={error}"
    )

    if context:

        message += (
            f" | CONTEXT={context}"
        )

    message += (
        "\n"
        + traceback_text
    )

    print(
        f"ERROR: {message}"
    )

    _append_log(
        ERROR_LOG_PATH,
        message,
    )









# ============================================================
# STOCK SYMBOLS
# ============================================================

def get_stock_symbols():
    """
    Download the stock symbol list from R2.

    Returns:
        list[str]: Stock symbols.
    """

    response = s3.get_object(
        Bucket=R2_BUCKET_NAME,
        Key=STOCK_SYMBOLS_PATH,
    )

    text = response["Body"].read().decode("utf-8")

    symbols = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return symbols


def get_completed_symbols():
    """
    Download the completed stock symbol list from R2.

    If the completed-symbol file does not exist yet,
    return an empty list.
    """

    try:
        response = s3.get_object(
            Bucket=R2_BUCKET_NAME,
            Key=COMPLETED_SYMBOLS_PATH,
        )

    except s3.exceptions.ClientError as error:
        error_code = error.response["Error"]["Code"]

        if error_code in ("404", "NoSuchKey"):
            return []

        raise

    text = response["Body"].read().decode("utf-8")

    symbols = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return symbols


def mark_stock_completed(symbol):
    """
    Add a stock symbol to the completed list and
    overwrite the completed list in R2.
    """

    completed_symbols = get_completed_symbols()

    if symbol not in completed_symbols:
        completed_symbols.append(symbol)

    text = "\n".join(completed_symbols)

    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=COMPLETED_SYMBOLS_PATH,
        Body=text.encode("utf-8"),
        ContentType="text/plain",
    )


# ============================================================
# STOCK DATA
# ============================================================

def download_stock_data(symbol):
    key = f"{symbol}.parquet"

    response = s3.get_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
    )

    data = response["Body"].read()

    return pd.read_parquet(io.BytesIO(data))


def save_input_data(symbol, input_data):
    ensure_r2_folder(INPUT_PATH)

    key = f"{INPUT_PATH}/{symbol}.parquet"

    buffer = io.BytesIO()

    input_data.to_parquet(
        buffer,
        index=False,
    )

    buffer.seek(0)

    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )

# ============================================================
# INPUT STATISTICS
# ============================================================

def save_input_statistics(symbol, statistics):
    ensure_r2_folder(INPUT_STATISTICS_PATH)

    key = f"{INPUT_STATISTICS_PATH}/{symbol}.parquet"

    buffer = io.BytesIO()

    statistics.to_parquet(
        buffer,
        index=False,
    )

    buffer.seek(0)

    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )


# ============================================================
# RISK / REWARD DATA
# ============================================================

def save_risk_reward_data(
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
    risk_reward_data,
):
    stock_folder = f"{RISK_REWARD_PATH}/{symbol}"
    trade_folder = f"{stock_folder}/{trade_type}"

    ensure_r2_folder(RISK_REWARD_PATH)
    ensure_r2_folder(stock_folder)
    ensure_r2_folder(trade_folder)

    filename = (
        f"sl_{stop_loss_percentage}_"
        f"rr_{risk_reward_ratio}.parquet"
    )

    key = f"{trade_folder}/{filename}"

    buffer = io.BytesIO()

    risk_reward_data.to_parquet(
        buffer,
        index=False,
    )

    buffer.seek(0)

    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream")





# ============================================================
# GRID DATA
# ============================================================

GRID_PATH = "grid"


def save_grid_configuration(
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
    dataset,
    grid_configuration,
):
    """
    Save the grid configuration for a specific
    stock / trade type / stop loss / risk reward strategy.

    Existing files are overwritten.
    """

    stock_folder = (
        f"{GRID_PATH}/{symbol}"
    )

    trade_folder = (
        f"{stock_folder}/{trade_type}"
    )

    strategy_folder = (
        f"{trade_folder}/"
        f"sl_{stop_loss_percentage}_"
        f"rr_{risk_reward_ratio}/"
        f"{dataset}"
    )

    ensure_r2_folder(GRID_PATH)
    ensure_r2_folder(stock_folder)
    ensure_r2_folder(trade_folder)
    ensure_r2_folder(strategy_folder)

    key = (
        f"{strategy_folder}/"
        f"grid_config.parquet"
    )

    buffer = io.BytesIO()

    grid_configuration.to_parquet(
        buffer,
        index=False,
    )

    buffer.seek(0)

    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )


def save_grid_data(
    symbol,
    trade_type,
    stop_loss_percentage,
    risk_reward_ratio,
    dataset,
    grid_data,
):
    """
    Save the populated grid cells for a specific
    stock / trade type / stop loss / risk reward strategy.

    Existing files are overwritten.
    """

    stock_folder = (
        f"{GRID_PATH}/{symbol}"
    )

    trade_folder = (
        f"{stock_folder}/{trade_type}"
    )

    strategy_folder = (
        f"{trade_folder}/"
        f"sl_{stop_loss_percentage}_"
        f"rr_{risk_reward_ratio}/"
        f"{dataset}"
    )

    ensure_r2_folder(GRID_PATH)
    ensure_r2_folder(stock_folder)
    ensure_r2_folder(trade_folder)
    ensure_r2_folder(strategy_folder)

    key = (
        f"{strategy_folder}/"
        f"grid_cells.parquet"
    )

    buffer = io.BytesIO()

    grid_data.to_parquet(
        buffer,
        index=False,
    )

    buffer.seek(0)

    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )
