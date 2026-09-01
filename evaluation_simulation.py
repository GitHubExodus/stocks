import io
import numpy as np
import pandas as pd

from numba import njit

from cloud_access import (
    get_stock_symbols,
    download_input_data,
    download_risk_reward_data,
    download_grid_configuration,
    download_grid_data,
    save_cross_stock_evaluation_equity_curve,
    log,
)


# ============================================================
# CONFIGURATION
# ============================================================

EVALUATION_SYMBOL = "AAPL"

WIN_RATE_THRESHOLD = 0.70

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
# TIMESTAMP MATCHING
# ============================================================

def match_rr_to_input_rows(
    rr_timestamps,
    input_timestamps,
):
    """
    Match every RR entry timestamp to the latest input
    timestamp inside the corresponding 5-minute candle.

    RR timestamp is the candle CLOSE.

    Example:

        RR timestamp = 1:00

        Search:
            12:55 <= input <= 1:00

        Preferred:
            1:00

        Otherwise:
            latest timestamp before 1:00
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

    input_ns = input_timestamps.asi8
    rr_ns = rr_timestamps.asi8

    five_minutes = pd.Timedelta(
        minutes=5
    ).value

    for rr_index, rr_end in enumerate(rr_ns):

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

            result[rr_index] = (
                right - 1
            )

    return result


# ============================================================
# PREPARE GRID
# ============================================================

def prepare_training_grid(
    grid_configuration,
    grid_data,
):
    """
    Prepare one stock's frozen training grid.
    """

    grid_configuration = (
        grid_configuration
        .copy()
        .reset_index(drop=True)
    )

    grid_data = (
        grid_data
        .copy()
        .reset_index(drop=True)
    )

    if len(grid_data) == 0:

        raise ValueError(
            "Cannot prepare an empty training grid."
        )

    input_names = (
        grid_configuration[
            "input_name"
        ]
        .tolist()
    )

    bin_counts = (
        grid_configuration[
            "bin_count"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    bin_sizes = (
        grid_configuration[
            "bin_size"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    mins = (
        grid_configuration[
            "min"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    num_grid_cells = len(grid_data)
    num_dimensions = len(bin_counts)

    # Verify that every expected dimension exists.
    required_columns = [
        f"dim_{dimension}"
        for dimension in range(num_dimensions)
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in grid_data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Grid data is missing coordinate columns: "
            + ", ".join(missing_columns)
        )

    grid_coordinates = np.empty(
        (
            num_grid_cells,
            num_dimensions,
        ),
        dtype=np.int64,
    )

    for dimension in range(num_dimensions):

        grid_coordinates[
            :,
            dimension
        ] = (
            grid_data[
                f"dim_{dimension}"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

    sort_keys = tuple(
        grid_coordinates[
            :,
            dimension
        ]
        for dimension in reversed(
            range(num_dimensions)
        )
    )

    sort_order = np.lexsort(
        sort_keys
    )

    grid_coordinates = (
        grid_coordinates[
            sort_order
        ]
    )

    total_trades = (
        grid_data[
            "Total_Trades"
        ]
        .to_numpy(
            dtype=np.float64
        )[sort_order]
    )

    total_wins = (
        grid_data[
            "Total_Wins"
        ]
        .to_numpy(
            dtype=np.float64
        )[sort_order]
    )

    return (
        input_names,
        bin_counts,
        bin_sizes,
        mins,
        grid_coordinates,
        total_trades,
        total_wins,
    )

# ============================================================
# NUMBA EVALUATION
# ============================================================

@njit
def _evaluate_grid_trades(
    input_values,
    input_valid,
    input_row_indices,
    rr_entry_prices,
    rr_exit_prices,
    grid_bin_counts,
    grid_bin_sizes,
    grid_mins,
    grid_coordinates,
    grid_total_trades,
    grid_total_wins,
    win_rate_threshold,
    is_long,
):
    """
    Evaluate AAPL trades using another stock's frozen grid.

    The grid decides whether a trade is selected.

    The actual dollar P&L comes from the AAPL
    entry_price and exit_price.

    Equity is additive:

        equity += dollar_return

    It does NOT compound.
    """

    num_rr = len(
        input_row_indices
    )

    num_dimensions = len(
        grid_bin_counts
    )

    num_grid_cells = len(
        grid_coordinates
    )

    selected_mask = np.zeros(
        num_rr,
        dtype=np.bool_,
    )

    equity = np.zeros(
        num_rr,
        dtype=np.float64,
    )

    current_equity = 0.0

    for rr_index in range(
        num_rr
    ):

        input_index = (
            input_row_indices[
                rr_index
            ]
        )

        if input_index < 0:
            continue

        # ====================================================
        # VALIDATE INPUT
        # ====================================================

        if not input_valid[
            input_index
        ]:
            continue

        # ====================================================
        # BUILD GRID COORDINATE
        # ====================================================

        coordinates = np.empty(
            num_dimensions,
            dtype=np.int64,
        )

        valid_coordinate = True

        for dimension in range(
            num_dimensions
        ):

            value = input_values[
                input_index,
                dimension,
            ]

            if not np.isfinite(
                value
            ):
                valid_coordinate = False
                break

            bin_count = (
                grid_bin_counts[
                    dimension
                ]
            )

            bin_size = (
                grid_bin_sizes[
                    dimension
                ]
            )

            minimum = (
                grid_mins[
                    dimension
                ]
            )

            if bin_count <= 0:
                valid_coordinate = False
                break

            if bin_size <= 0.0:

                coordinate = 0

            else:

                coordinate = int(
                    np.floor(
                        (
                            value
                            - minimum
                        )
                        / bin_size
                    )
                )

                if coordinate < 0:

                    coordinate = 0

                elif coordinate >= bin_count:

                    coordinate = (
                        bin_count - 1
                    )

            coordinates[
                dimension
            ] = coordinate

        if not valid_coordinate:
            continue

        # ====================================================
        # BINARY SEARCH GRID
        # ====================================================

        left = 0
        right = num_grid_cells

        while left < right:

            middle = (
                left
                + (
                    right - left
                ) // 2
            )

            training_less = False

            for dimension in range(
                num_dimensions
            ):

                training_value = (
                    grid_coordinates[
                        middle,
                        dimension,
                    ]
                )

                target_value = (
                    coordinates[
                        dimension
                    ]
                )

                if (
                    training_value
                    < target_value
                ):

                    training_less = True
                    break

                elif (
                    training_value
                    > target_value
                ):

                    break

            if training_less:

                left = middle + 1

            else:

                right = middle

        cell_index = left

        # ====================================================
        # VERIFY EXACT MATCH
        # ====================================================

        if cell_index >= num_grid_cells:
            continue

        exact_match = True

        for dimension in range(
            num_dimensions
        ):

            if (
                grid_coordinates[
                    cell_index,
                    dimension,
                ]
                != coordinates[
                    dimension
                ]
            ):

                exact_match = False
                break

        if not exact_match:
            continue

        # ====================================================
        # GRID STATISTICS
        # ====================================================

        total_trades = (
            grid_total_trades[
                cell_index
            ]
        )

        total_wins = (
            grid_total_wins[
                cell_index
            ]
        )

        if total_trades <= 0.0:
            continue

        training_win_rate = (
            total_wins
            / total_trades
        )

        # ====================================================
        # WIN RATE FILTER
        # ====================================================

        if (
            training_win_rate
            <= win_rate_threshold
        ):
            continue

        # ====================================================
        # AAPL ENTRY / EXIT
        # ====================================================

        entry_price = (
            rr_entry_prices[
                rr_index
            ]
        )

        exit_price = (
            rr_exit_prices[
                rr_index
            ]
        )

        if not np.isfinite(
            entry_price
        ):
            continue

        if not np.isfinite(
            exit_price
        ):
            continue

        if entry_price <= 0.0:
            continue

        # ====================================================
        # ACTUAL DOLLAR P&L
        # ====================================================

        if is_long:

            dollar_return = (
                exit_price
                - entry_price
            )

        else:

            dollar_return = (
                entry_price
                - exit_price
            )

        if not np.isfinite(
            dollar_return
        ):
            continue

        # ====================================================
        # ADDITIVE EQUITY
        # ====================================================

        current_equity += (
            dollar_return
        )

        selected_mask[
            rr_index
        ] = True

        equity[
            rr_index
        ] = current_equity

    return (
        selected_mask,
        equity,
    )


# ============================================================
# EVALUATE ONE GRID AGAINST AAPL
# ============================================================

def evaluate_grid(
    input_data,
    risk_reward_data,
    grid_configuration,
    grid_data,
    trade_type,
):
    """
    Evaluate EVALUATION_SYMBOL's trades against one
    stock's frozen training grid.

    The evaluation stock provides:
        - input values
        - entry prices
        - exit prices

    The grid stock provides:
        - grid configuration
        - grid cell statistics
    """

    # ========================================================
    # EMPTY GRID
    # ========================================================

    if grid_data is None or len(grid_data) == 0:

        return pd.DataFrame(
            columns=[
                "start_timestamp",
                "end_timestamp",
                "time_elapsed_minutes",
                "entry_price",
                "exit_price",
                "dollar_return",
                "equity",
            ]
        )

    # ========================================================
    # INPUT DATA
    # ========================================================

    input_data = input_data.copy()

    input_data["timestamp"] = (
        pd.to_datetime(
            input_data["timestamp"],
            utc=True,
        )
        .dt.tz_convert(
            "America/New_York"
        )
    )

    input_data = (
        input_data
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ========================================================
    # RISK / REWARD DATA
    # ========================================================

    risk_reward_data = risk_reward_data.copy()

    risk_reward_data["start_timestamp"] = (
        pd.to_datetime(
            risk_reward_data["start_timestamp"],
            utc=True,
        )
        .dt.tz_convert(
            "America/New_York"
        )
    )

    if "end_timestamp" in risk_reward_data.columns:

        risk_reward_data["end_timestamp"] = (
            pd.to_datetime(
                risk_reward_data["end_timestamp"],
                utc=True,
            )
            .dt.tz_convert(
                "America/New_York"
            )
        )

    risk_reward_data = (
        risk_reward_data
        .sort_values("start_timestamp")
        .reset_index(drop=True)
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
    # PREPARE GRID
    # ========================================================

    (
        input_names,
        bin_counts,
        bin_sizes,
        mins,
        grid_coordinates,
        grid_total_trades,
        grid_total_wins,
    ) = prepare_training_grid(
        grid_configuration=grid_configuration,
        grid_data=grid_data,
    )

    # ========================================================
    # VERIFY INPUT FEATURES EXIST
    # ========================================================

    missing_inputs = [
        name
        for name in input_names
        if name not in input_data.columns
    ]

    if missing_inputs:

        raise ValueError(
            "Missing evaluation input columns: "
            + ", ".join(missing_inputs)
        )

    # ========================================================
    # INPUT MATRIX
    # ========================================================

    input_values = (
        input_data[
            input_names
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    input_valid = np.all(
        np.isfinite(input_values),
        axis=1,
    )

    # ========================================================
    # RR PRICES
    # ========================================================

    rr_entry_prices = (
        risk_reward_data[
            "entry_price"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    rr_exit_prices = (
        risk_reward_data[
            "exit_price"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    rr_time_elapsed = (
        risk_reward_data[
            "time_elapsed_minutes"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # ========================================================
    # NUMBA EVALUATION
    # ========================================================

    (
        selected_mask,
        equity,
    ) = _evaluate_grid_trades(
        input_values=input_values,
        input_valid=input_valid,
        input_row_indices=input_row_indices,
        rr_entry_prices=rr_entry_prices,
        rr_exit_prices=rr_exit_prices,
        grid_bin_counts=bin_counts,
        grid_bin_sizes=bin_sizes,
        grid_mins=mins,
        grid_coordinates=grid_coordinates,
        grid_total_trades=grid_total_trades,
        grid_total_wins=grid_total_wins,
        win_rate_threshold=WIN_RATE_THRESHOLD,
        is_long=(
            trade_type == "long"
        ),
    )

    # ========================================================
    # SELECTED TRADES
    # ========================================================

    selected_indices = np.flatnonzero(
        selected_mask
    )

    if len(selected_indices) == 0:

        return pd.DataFrame(
            columns=[
                "start_timestamp",
                "end_timestamp",
                "time_elapsed_minutes",
                "entry_price",
                "exit_price",
                "dollar_return",
                "equity",
            ]
        )

    # ========================================================
    # SELECT RR ROWS
    # ========================================================

    selected_rr = (
        risk_reward_data
        .iloc[selected_indices]
        .reset_index(drop=True)
    )

    selected_entry = (
        rr_entry_prices[
            selected_indices
        ]
    )

    selected_exit = (
        rr_exit_prices[
            selected_indices
        ]
    )

    # ========================================================
    # ACTUAL EVALUATION P&L
    # ========================================================

    if trade_type == "long":

        dollar_returns = (
            selected_exit
            - selected_entry
        )

    else:

        dollar_returns = (
            selected_entry
            - selected_exit
        )

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    equity_curve = pd.DataFrame(
        {
            "start_timestamp":
                selected_rr[
                    "start_timestamp"
                ].to_numpy(),

            "end_timestamp":
                selected_rr[
                    "end_timestamp"
                ].to_numpy(),

            "time_elapsed_minutes":
                rr_time_elapsed[
                    selected_indices
                ],

            "entry_price":
                selected_entry,

            "exit_price":
                selected_exit,

            "dollar_return":
                dollar_returns,

            "equity":
                equity[
                    selected_indices
                ],
        }
    )

    return equity_curve

# ============================================================
# MAIN
# ============================================================

def main():

    log(
        f"EVALUATION STARTED | "
        f"EVALUATION_SYMBOL={EVALUATION_SYMBOL}"
    )

    # ========================================================
    # GET ALL STOCKS
    # ========================================================

    symbols = get_stock_symbols()

    symbols = [
        str(symbol).strip().upper()
        for symbol in symbols
        if str(symbol).strip()
    ]

    log(
        f"Evaluation grid stocks loaded | "
        f"count={len(symbols):,}"
    )

    # ========================================================
    # DOWNLOAD AAPL INPUT DATA ONCE
    # ========================================================

    log(
        f"Downloading evaluation input data | "
        f"symbol={EVALUATION_SYMBOL}"
    )

    input_data = (
        download_input_data(
            EVALUATION_SYMBOL
        )
    )

    log(
        f"Evaluation input data downloaded | "
        f"symbol={EVALUATION_SYMBOL} | "
        f"rows={len(input_data):,}"
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
                    f"STRATEGY START | "
                    f"evaluation_symbol={EVALUATION_SYMBOL} | "
                    f"trade_type={trade_type} | "
                    f"SL={stop_loss_percentage} | "
                    f"RR={risk_reward_ratio}"
                )

                # =================================================
                # DOWNLOAD AAPL RR DATA
                #
                # These are the trades being evaluated.
                # =================================================

                risk_reward_data = (
                    download_risk_reward_data(
                        symbol=(
                            EVALUATION_SYMBOL
                        ),
                        trade_type=(
                            trade_type
                        ),
                        stop_loss_percentage=(
                            stop_loss_percentage
                        ),
                        risk_reward_ratio=(
                            risk_reward_ratio
                        ),
                    )
                )

                log(
                    f"AAPL RR data downloaded | "
                    f"rows={len(risk_reward_data):,}"
                )

                # =================================================
                # EVERY STOCK'S GRID
                # =================================================

                for grid_symbol in symbols:

                    log(
                        f"GRID EVALUATION START | "
                        f"evaluation_symbol="
                        f"{EVALUATION_SYMBOL} | "
                        f"grid_symbol={grid_symbol} | "
                        f"trade_type={trade_type} | "
                        f"SL={stop_loss_percentage} | "
                        f"RR={risk_reward_ratio}"
                    )

                    try:

                        # =========================================
                        # DOWNLOAD GRID CONFIGURATION
                        # =========================================

                        grid_configuration = (
                            download_grid_configuration(
                                symbol=(
                                    grid_symbol
                                ),
                                trade_type=(
                                    trade_type
                                ),
                                stop_loss_percentage=(
                                    stop_loss_percentage
                                ),
                                risk_reward_ratio=(
                                    risk_reward_ratio
                                ),
                                dataset="training",
                            )
                        )

                        # =========================================
                        # DOWNLOAD GRID CELLS
                        # =========================================

                        grid_data = (
                            download_grid_data(
                                symbol=(
                                    grid_symbol
                                ),
                                trade_type=(
                                    trade_type
                                ),
                                stop_loss_percentage=(
                                    stop_loss_percentage
                                ),
                                risk_reward_ratio=(
                                    risk_reward_ratio
                                ),
                                dataset="training",
                            )
                        )

                        log(
                            f"Grid downloaded | "
                            f"grid_symbol={grid_symbol} | "
                            f"cells={len(grid_data):,}"
                        )


                        # =========================================================
                        # SKIP EMPTY GRID
                        # =========================================================

                        if grid_data is None or len(grid_data) == 0:

                            log(
                                f"GRID EVALUATION SKIPPED | "
                                f"evaluation_symbol={EVALUATION_SYMBOL} | "
                                f"grid_symbol={grid_symbol} | "
                                f"trade_type={trade_type} | "
                                f"SL={stop_loss_percentage} | "
                                f"RR={risk_reward_ratio} | "
                                f"reason=empty_grid"
                            )

                            continue

                        # =========================================
                        # EVALUATE AAPL AGAINST THIS GRID
                        # =========================================

                        equity_curve = (
                            evaluate_grid(
                                input_data=(
                                    input_data
                                ),
                                risk_reward_data=(
                                    risk_reward_data
                                ),
                                grid_configuration=(
                                    grid_configuration
                                ),
                                grid_data=(
                                    grid_data
                                ),
                                trade_type=(
                                    trade_type
                                ),
                            )
                        )

                        # =========================================
                        # SAVE
                        # =========================================

                        save_cross_stock_evaluation_equity_curve(
                            evaluation_symbol=(
                                EVALUATION_SYMBOL
                            ),
                            grid_symbol=(
                                grid_symbol
                            ),
                            trade_type=(
                                trade_type
                            ),
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

                        log(
                            f"GRID EVALUATION COMPLETE | "
                            f"evaluation_symbol="
                            f"{EVALUATION_SYMBOL} | "
                            f"grid_symbol={grid_symbol} | "
                            f"trades={len(equity_curve):,}"
                        )

                    except Exception as error:

                        log(
                            f"GRID EVALUATION FAILED | "
                            f"evaluation_symbol="
                            f"{EVALUATION_SYMBOL} | "
                            f"grid_symbol={grid_symbol} | "
                            f"trade_type={trade_type} | "
                            f"SL={stop_loss_percentage} | "
                            f"RR={risk_reward_ratio} | "
                            f"error={error}"
                        )

                        raise

                log(
                    f"STRATEGY COMPLETE | "
                    f"evaluation_symbol={EVALUATION_SYMBOL} | "
                    f"trade_type={trade_type} | "
                    f"SL={stop_loss_percentage} | "
                    f"RR={risk_reward_ratio}"
                )

    log(
        f"EVALUATION COMPLETE | "
        f"EVALUATION_SYMBOL={EVALUATION_SYMBOL}"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
