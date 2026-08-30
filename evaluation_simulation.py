
import numpy as np
import pandas as pd

from numba import njit

from cloud_access import (
    download_input_data,
    download_risk_reward_data,
    download_grid_configuration,
    download_grid_data,
    save_evaluation_equity_curve,
    log,
)


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL = "DELL"

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
    Match every RR timestamp to the latest input timestamp
    inside the corresponding 5-minute candle.

    IMPORTANT:

        RR timestamp = candle CLOSE.

    Therefore:

        RR timestamp = 1:00

        Search interval:

            12:55 <= input timestamp <= 1:00

        Preferred match:

            1:00

        If 1:00 does not exist:

            latest available timestamp before 1:00

    Example:

        RR timestamp:
            1:00

        Input:
            12:55
            12:57
            12:59
            1:00

        Selected:
            1:00

    If 1:00 does not exist:

        Input:
            12:55
            12:57
            12:59

        Selected:
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
        input_timestamps
        .asi8
    )

    rr_ns = (
        rr_timestamps
        .asi8
    )

    for rr_index, rr_end in enumerate(
        rr_ns
    ):

        rr_start = (
            rr_end
            - pd.Timedelta(
                minutes=5
            ).value
        )

        # ----------------------------------------------------
        # Include the RR timestamp itself.
        #
        # rr_start <= input <= rr_end
        # ----------------------------------------------------

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
# NUMBA GRID EVALUATION
# ============================================================

# ============================================================
# NUMBA GRID EVALUATION
# ============================================================

@njit
def _evaluate_grid_trades(
    input_values,
    input_valid,
    input_row_indices,
    rr_returns,
    rr_time_elapsed,
    grid_bin_counts,
    grid_bin_sizes,
    grid_mins,
    grid_coordinates,
    grid_total_trades,
    grid_total_wins,
    win_rate_threshold,
):
    """
    Evaluate RR trades against the FROZEN TRAINING GRID.

    ONLY the training grid is used.

    Grid lookup uses direct coordinate comparison.

    No flattened integer is created.

    Therefore there is no integer overflow.
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

    equity = np.ones(
        num_rr,
        dtype=np.float64,
    )

    current_equity = 1.0

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
        # VALIDATE INPUT ROW
        # ====================================================

        if not input_valid[
            input_index
        ]:
            continue

        # ====================================================
        # CALCULATE GRID COORDINATE
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
                        bin_count
                        - 1
                    )

            coordinates[
                dimension
            ] = coordinate

        if not valid_coordinate:
            continue

        # ====================================================
        # BINARY SEARCH
        #
        # Search the sorted TRAINING GRID coordinates.
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

            # ------------------------------------------------
            # Lexicographic comparison:
            #
            # training coordinate < target coordinate
            # ------------------------------------------------

            training_less = False
            training_greater = False

            for dimension in range(
                num_dimensions
            ):

                training_value = (
                    grid_coordinates[
                        middle,
                        dimension
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

                    training_greater = True
                    break

            if training_less:

                left = (
                    middle + 1
                )

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
                    dimension
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
        # TRAINING GRID STATISTICS
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
        # TRAINING GRID FILTER
        # ====================================================

        if (
            training_win_rate
            <= win_rate_threshold
        ):
            continue

        # ====================================================
        # RR DATA
        # ====================================================

        trade_return = (
            rr_returns[
                rr_index
            ]
        )

        elapsed_time = (
            rr_time_elapsed[
                rr_index
            ]
        )

        if not np.isfinite(
            trade_return
        ):
            continue

        if not np.isfinite(
            elapsed_time
        ):
            continue

        # ====================================================
        # COMPOUND EQUITY
        # ====================================================

        current_equity *= (
            1.0
            + trade_return / 100.0
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
# PREPARE TRAINING GRID
# ============================================================

def prepare_training_grid(
    grid_configuration,
    grid_data,
):
    """
    Prepare the FROZEN TRAINING GRID for fast Numba lookup.

    ONLY the training grid is used.

    No grid is created.
    No grid is modified.

    Grid coordinates are stored directly instead of being
    flattened into a potentially overflowing integer.
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

    # ========================================================
    # GRID CONFIGURATION
    # ========================================================

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

    # ========================================================
    # GRID COORDINATES
    # ========================================================

    num_grid_cells = len(
        grid_data
    )

    num_dimensions = len(
        bin_counts
    )

    grid_coordinates = np.empty(
        (
            num_grid_cells,
            num_dimensions,
        ),
        dtype=np.int64,
    )

    for dimension in range(
        num_dimensions
    ):

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

    # ========================================================
    # SORT GRID COORDINATES
    #
    # np.lexsort sorts by the last key first.
    #
    # This gives us:
    #
    # dim_0
    # dim_1
    # dim_2
    # ...
    #
    # lexicographic ordering.
    # ========================================================

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

    # ========================================================
    # TRAINING GRID STATISTICS
    # ========================================================

    total_trades = (
        grid_data[
            "Total_Trades"
        ]
        .to_numpy(
            dtype=np.float64
        )[
            sort_order
        ]
    )

    total_wins = (
        grid_data[
            "Total_Wins"
        ]
        .to_numpy(
            dtype=np.float64
        )[
            sort_order
        ]
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
# EVALUATE ONE STRATEGY
# ============================================================

def evaluate_strategy(
    input_data,
    risk_reward_data,
    grid_configuration,
    grid_data,
):
    """
    Evaluate one:

        trade_type
        stop loss
        RR

    combination against the FROZEN TRAINING GRID.
    """

    # ========================================================
    # PREPARE INPUT DATA
    # ========================================================

    input_data = (
        input_data
        .copy()
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

    # ========================================================
    # PREPARE RR DATA
    # ========================================================

    risk_reward_data = (
        risk_reward_data
        .copy()
    )

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

    risk_reward_data = (
        risk_reward_data
        .sort_values(
            "start_timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # MATCH RR CANDLES TO INPUT ROWS
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
    # PREPARE TRAINING GRID
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
    # BUILD INPUT NUMPY MATRIX
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
        np.isfinite(
            input_values
        ),
        axis=1,
    )

    # ========================================================
    # RR NUMPY ARRAYS
    # ========================================================

    rr_returns = (
        risk_reward_data[
            "trade_return_percent"
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
        rr_returns=rr_returns,
        rr_time_elapsed=rr_time_elapsed,
        grid_bin_counts=bin_counts,
        grid_bin_sizes=bin_sizes,
        grid_mins=mins,
        grid_coordinates=grid_coordinates,
        grid_total_trades=grid_total_trades,
        grid_total_wins=grid_total_wins,
        win_rate_threshold=WIN_RATE_THRESHOLD,
    )

    # ========================================================
    # CREATE EQUITY CURVE
    # ========================================================

    selected_indices = np.flatnonzero(
        selected_mask
    )

    if len(
        selected_indices
    ) == 0:

        return pd.DataFrame(
            columns=[
                "start_timestamp",
                "time_elapsed_minutes",
                "trade_return_percent",
                "equity",
            ]
        )

    equity_curve = pd.DataFrame(
        {
            "start_timestamp":
                risk_reward_data[
                    "start_timestamp"
                ].iloc[
                    selected_indices
                ].to_numpy(),

            "time_elapsed_minutes":
                rr_time_elapsed[
                    selected_indices
                ],

            "trade_return_percent":
                rr_returns[
                    selected_indices
                ],

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
        f"SYMBOL={SYMBOL}"
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
        f"symbol={SYMBOL} | "
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
                    f"EVALUATION STRATEGY START | "
                    f"symbol={SYMBOL} | "
                    f"trade_type={trade_type} | "
                    f"SL={stop_loss_percentage} | "
                    f"RR={risk_reward_ratio}"
                )

                # =================================================
                # DOWNLOAD RR DATA
                # =================================================

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

                # =================================================
                # DOWNLOAD TRAINING GRID CONFIGURATION
                #
                # ONLY TRAINING IS USED.
                # =================================================

                grid_configuration = (
                    download_grid_configuration(
                        symbol=SYMBOL,
                        trade_type=trade_type,
                        stop_loss_percentage=(
                            stop_loss_percentage
                        ),
                        risk_reward_ratio=(
                            risk_reward_ratio
                        ),
                        dataset="training",
                    )
                )

                # =================================================
                # DOWNLOAD TRAINING GRID CELLS
                #
                # ONLY TRAINING IS USED.
                # =================================================

                grid_data = (
                    download_grid_data(
                        symbol=SYMBOL,
                        trade_type=trade_type,
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
                    f"Training grid downloaded | "
                    f"symbol={SYMBOL} | "
                    f"trade_type={trade_type} | "
                    f"SL={stop_loss_percentage} | "
                    f"RR={risk_reward_ratio} | "
                    f"cells={len(grid_data):,}"
                )

                # =================================================
                # EVALUATE
                # =================================================

                equity_curve = (
                    evaluate_strategy(
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
                    )
                )

                # =================================================
                # SAVE
                # =================================================

                save_evaluation_equity_curve(
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

                log(
                    f"EVALUATION STRATEGY COMPLETE | "
                    f"symbol={SYMBOL} | "
                    f"trade_type={trade_type} | "
                    f"SL={stop_loss_percentage} | "
                    f"RR={risk_reward_ratio} | "
                    f"trades={len(equity_curve):,}"
                )

    log(
        f"EVALUATION COMPLETE | "
        f"SYMBOL={SYMBOL}"
    )


if __name__ == "__main__":
    main()
