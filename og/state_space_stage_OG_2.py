import numpy as np
import pandas as pd
from numba import njit

from statistics_api import StatisticsAPI
from grid_api import GridTable

from cloud_access import (
    save_input_statistics,
    save_grid_configuration,
    save_grid_data,
    log,
)


class StateSpaceStage:

    def __init__(self):
        self.input_data = None
        self.input_statistics = None
        self.grid_table = None
        self.grid_configuration = None
        self.grid_data = None

    # ============================================================
    # MAIN STAGE
    # ============================================================

    def run(
        self,
        symbol,
        trade_type,
        stop_loss_percentage,
        risk_reward_ratio,
        input_data,
        input_statistics,
        risk_reward_data,
        bin_counts,
    ):

        log(
            f"State Space Stage started | "
            f"symbol={symbol} | "
            f"trade_type={trade_type} | "
            f"SL={stop_loss_percentage} | "
            f"RR={risk_reward_ratio}"
        )

        # ========================================================
        # 1. LOAD
        # ========================================================

        self.input_data = input_data.copy()
        self.input_statistics = (
            input_statistics.copy()
        )

        # ========================================================
        # 2. VALIDATE BIN COUNTS
        # ========================================================

        input_names = (
            self.input_statistics[
                "input"
            ].tolist()
        )

        missing_bin_counts = [
            input_name
            for input_name in input_names
            if input_name not in bin_counts
        ]

        if missing_bin_counts:
            raise KeyError(
                "Missing bin counts for inputs: "
                + ", ".join(
                    missing_bin_counts
                )
            )

        # ========================================================
        # 3. CALCULATE BIN CONFIGURATION
        # ========================================================

        bin_counts_final = []
        bin_sizes_final = []

        for index, row in (
            self.input_statistics.iterrows()
        ):

            configured_bin_count = (
                bin_counts[
                    row["input"]
                ]
            )

            result = (
                StatisticsAPI
                .calculate_bin_configuration(
                    minimum=row["min"],
                    q1=row["q1"],
                    q3=row["q3"],
                    maximum=row["max"],
                    configured_bin_count=(
                        configured_bin_count
                    ),
                )
            )

            bin_counts_final.append(
                result["bin_count"]
            )

            bin_sizes_final.append(
                result["bin_size"]
            )

            self.input_statistics.loc[
                index,
                "bin_count"
            ] = result["bin_count"]

            self.input_statistics.loc[
                index,
                "bin_size"
            ] = result["bin_size"]

        # ========================================================
        # 4. SAVE UPDATED INPUT STATISTICS
        # ========================================================

        save_input_statistics(
            symbol,
            self.input_statistics,
        )

        # ========================================================
        # 5. CREATE GRID CONFIGURATION
        # ========================================================

        mins = (
            self.input_statistics[
                "min"
            ].to_numpy(
                dtype=np.float64
            )
        )

        q1s = (
            self.input_statistics[
                "q1"
            ].to_numpy(
                dtype=np.float64
            )
        )

        medians = (
            self.input_statistics[
                "median"
            ].to_numpy(
                dtype=np.float64
            )
        )

        q3s = (
            self.input_statistics[
                "q3"
            ].to_numpy(
                dtype=np.float64
            )
        )

        maxs = (
            self.input_statistics[
                "max"
            ].to_numpy(
                dtype=np.float64
            )
        )

        bin_counts_final = np.asarray(
            bin_counts_final,
            dtype=np.int64,
        )

        bin_sizes_final = np.asarray(
            bin_sizes_final,
            dtype=np.float64,
        )

        # ========================================================
        # 6. CREATE GRID TABLES
        # ========================================================

        training_grid_table = GridTable(
            input_names=input_names,
            bin_counts=bin_counts_final,
            bin_sizes=bin_sizes_final,
            mins=mins,
            q1s=q1s,
            medians=medians,
            q3s=q3s,
            maxs=maxs,
        )

        validation_grid_table = GridTable(
            input_names=input_names,
            bin_counts=bin_counts_final,
            bin_sizes=bin_sizes_final,
            mins=mins,
            q1s=q1s,
            medians=medians,
            q3s=q3s,
            maxs=maxs,
        )

        # ========================================================
        # 7. PREPARE TIMESTAMPS
        # ========================================================

        validation_start = pd.Timestamp(
            "2025-01-01",
            tz="America/New_York",
        )

        # --------------------------------------------------------
        # RR timestamps are already New York time.
        #
        # Do NOT convert back to UTC.
        # --------------------------------------------------------

        risk_reward_data = (
            risk_reward_data
            .sort_values("start_timestamp")
            .reset_index(drop=True)
        )

        risk_reward_timestamps = (
            pd.to_datetime(
                risk_reward_data[
                    "start_timestamp"
                ]
            )
        )

        # Make sure they are timezone-aware.
        if (
            risk_reward_timestamps.dt.tz
            is None
        ):
            risk_reward_timestamps = (
                risk_reward_timestamps.dt.tz_localize(
                    "America/New_York"
                )
            )
        else:
            risk_reward_timestamps = (
                risk_reward_timestamps.dt.tz_convert(
                    "America/New_York"
                )
            )

        # ========================================================
        # 8. CREATE INPUT TIMESTAMP LOOKUP
        # ========================================================

        input_timestamps = pd.to_datetime(
            self.input_data[
                "timestamp"
            ]
        )

        if input_timestamps.dt.tz is None:
            input_timestamps = (
                input_timestamps.dt.tz_localize(
                    "America/New_York"
                )
            )
        else:
            input_timestamps = (
                input_timestamps.dt.tz_convert(
                    "America/New_York"
                )
            )

        input_timestamp_lookup = (
            self.input_data.copy()
        )

        input_timestamp_lookup[
            "timestamp"
        ] = input_timestamps

        # --------------------------------------------------------
        # Duplicate timestamps:
        #
        # Keep the first input row for each timestamp.
        # --------------------------------------------------------

        input_timestamp_lookup = (
        input_timestamp_lookup
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .set_index("timestamp")
        .sort_index()
    )

        # ========================================================
        # 9. CONVERT INPUT DATA TO NUMPY
        # ========================================================

        input_values = (
            input_timestamp_lookup[
                input_names
            ].to_numpy(
                dtype=np.float64
            )
        )

        input_lookup_timestamps = (
            input_timestamp_lookup.index
            .view("int64")
        )

        rr_timestamp_values = (
            risk_reward_timestamps
            .astype("int64")
            .to_numpy()
        )

        # ========================================================
        # 10. SPLIT TRAINING / VALIDATION
        # ========================================================

        validation_start_value = (
            validation_start.value
        )

        training_mask = (
            rr_timestamp_values
            < validation_start_value
        )

        validation_mask = (
            rr_timestamp_values
            >= validation_start_value
        )

        training_indices = np.flatnonzero(
            training_mask
        )

        validation_indices = np.flatnonzero(
            validation_mask
        )

        # ========================================================
        # 11. PREPARE RR STATISTICS
        # ========================================================

        rr_time_elapsed = (
            risk_reward_data[
                "time_elapsed_minutes"
            ].to_numpy(
                dtype=np.float64
            )
        )

        rr_returns = (
            risk_reward_data[
                "trade_return_percent"
            ].to_numpy(
                dtype=np.float64
            )
        )

        # ========================================================
        # 12. MAP RR TIMESTAMPS TO INPUT ROWS
        # ========================================================

        input_row_indices = (
            _match_timestamps(
                rr_timestamp_values,
                input_lookup_timestamps,
            )
        )

        # ========================================================
        # 13. CALCULATE GRID COORDINATES
        # ========================================================

        coordinates = (
            _calculate_grid_coordinates(
                input_values=input_values,
                input_row_indices=(
                    input_row_indices
                ),
                bin_counts=bin_counts_final,
                bin_sizes=bin_sizes_final,
                mins=mins,
            )
        )

        # ========================================================
        # 14. POPULATE TRAINING GRID
        # ========================================================

        training_added = 0

        for rr_index in training_indices:

            input_row_index = (
                input_row_indices[
                    rr_index
                ]
            )

            if input_row_index < 0:
                continue

            coordinate_array = coordinates[
                rr_index
            ]

            if np.any(
                coordinate_array < 0
            ):
                continue

            coordinate = tuple(
                coordinate_array
            )

            time_elapsed = (
                rr_time_elapsed[
                    rr_index
                ]
            )

            trade_return = (
                rr_returns[
                    rr_index
                ]
            )

            if (
                not np.isfinite(
                    time_elapsed
                )
                or not np.isfinite(
                    trade_return
                )
            ):
                continue

            training_grid_table.add_state(
                coordinate,
                {
                    "time_elapsed_minutes":
                        time_elapsed,

                    "trade_return_percent":
                        trade_return,
                },
            )

            training_added += 1

        # ========================================================
        # 15. POPULATE VALIDATION GRID
        # ========================================================

        validation_added = 0

        for rr_index in validation_indices:

            input_row_index = (
                input_row_indices[
                    rr_index
                ]
            )

            if input_row_index < 0:
                continue

            coordinate_array = coordinates[
                rr_index
            ]

            if np.any(
                coordinate_array < 0
            ):
                continue

            coordinate = tuple(
                coordinate_array
            )

            time_elapsed = (
                rr_time_elapsed[
                    rr_index
                ]
            )

            trade_return = (
                rr_returns[
                    rr_index
                ]
            )

            if (
                not np.isfinite(
                    time_elapsed
                )
                or not np.isfinite(
                    trade_return
                )
            ):
                continue

            validation_grid_table.add_state(
                coordinate,
                {
                    "time_elapsed_minutes":
                        time_elapsed,

                    "trade_return_percent":
                        trade_return,
                },
            )

            validation_added += 1

        # ========================================================
        # 16. GET GRID DATA
        # ========================================================

        training_grid_configuration = (
            training_grid_table
            .get_grid_configuration()
        )

        training_grid_data = (
            training_grid_table
            .get_grid_data()
        )

        validation_grid_configuration = (
            validation_grid_table
            .get_grid_configuration()
        )

        validation_grid_data = (
            validation_grid_table
            .get_grid_data()
        )

        # ========================================================
        # 17. SAVE TRAINING GRID
        # ========================================================

        save_grid_configuration(
            symbol=symbol,
            trade_type=trade_type,
            stop_loss_percentage=(
                stop_loss_percentage
            ),
            risk_reward_ratio=(
                risk_reward_ratio
            ),
            dataset="training",
            grid_configuration=(
                training_grid_configuration
            ),
        )

        save_grid_data(
            symbol=symbol,
            trade_type=trade_type,
            stop_loss_percentage=(
                stop_loss_percentage
            ),
            risk_reward_ratio=(
                risk_reward_ratio
            ),
            dataset="training",
            grid_data=training_grid_data,
        )

        # ========================================================
        # 18. SAVE VALIDATION GRID
        # ========================================================

        save_grid_configuration(
            symbol=symbol,
            trade_type=trade_type,
            stop_loss_percentage=(
                stop_loss_percentage
            ),
            risk_reward_ratio=(
                risk_reward_ratio
            ),
            dataset="validation",
            grid_configuration=(
                validation_grid_configuration
            ),
        )

        save_grid_data(
            symbol=symbol,
            trade_type=trade_type,
            stop_loss_percentage=(
                stop_loss_percentage
            ),
            risk_reward_ratio=(
                risk_reward_ratio
            ),
            dataset="validation",
            grid_data=validation_grid_data,
        )

        # ========================================================
        # 19. COMPLETE
        # ========================================================

        log(
            f"State Space Stage completed | "
            f"symbol={symbol} | "
            f"trade_type={trade_type} | "
            f"SL={stop_loss_percentage} | "
            f"RR={risk_reward_ratio} | "
            f"training_trades={training_added:,} | "
            f"validation_trades={validation_added:,} | "
            f"training_cells="
            f"{len(training_grid_data):,} | "
            f"validation_cells="
            f"{len(validation_grid_data):,}"
        )

        # ========================================================
        # 20. RETURN
        # ========================================================

        return {
            "training": {
                "grid_configuration":
                    training_grid_configuration,

                "grid_data":
                    training_grid_data,
            },

            "validation": {
                "grid_configuration":
                    validation_grid_configuration,

                "grid_data":
                    validation_grid_data,
            },

            "input_statistics":
                self.input_statistics,
        }


# =================================================================
# NUMPY / NUMBA HELPERS
# =================================================================

@njit
def _match_timestamps(
    rr_timestamps,
    input_timestamps,
):
    """
    Find the exact input-data row corresponding
    to every RR start timestamp.

    Returns -1 when no matching timestamp exists.

    Both timestamp arrays must be sorted.
    """

    result = np.full(
        len(rr_timestamps),
        -1,
        dtype=np.int64,
    )

    input_position = 0
    input_length = len(
        input_timestamps
    )

    for rr_position in range(
        len(rr_timestamps)
    ):

        rr_timestamp = (
            rr_timestamps[
                rr_position
            ]
        )

        while (
            input_position
            < input_length
            and input_timestamps[
                input_position
            ]
            < rr_timestamp
        ):
            input_position += 1

        if (
            input_position
            < input_length
            and input_timestamps[
                input_position
            ]
            == rr_timestamp
        ):
            result[
                rr_position
            ] = input_position

    return result


@njit
def _calculate_grid_coordinates(
    input_values,
    input_row_indices,
    bin_counts,
    bin_sizes,
    mins,
):
    """
    Calculate grid coordinates for every RR trade.

    A coordinate of -1 in every dimension means
    that the corresponding RR row cannot be used.
    """

    num_rows = len(
        input_row_indices
    )

    num_dimensions = len(
        bin_counts
    )

    coordinates = np.full(
        (
            num_rows,
            num_dimensions,
        ),
        -1,
        dtype=np.int64,
    )

    for row_index in range(
        num_rows
    ):

        input_row = (
            input_row_indices[
                row_index
            ]
        )

        if input_row < 0:
            continue

        valid = True

        for dimension in range(
            num_dimensions
        ):

            value = input_values[
                input_row,
                dimension,
            ]

            if not np.isfinite(
                value
            ):
                valid = False
                break

            bin_count = (
                bin_counts[
                    dimension
                ]
            )

            bin_size = (
                bin_sizes[
                    dimension
                ]
            )

            minimum = (
                mins[
                    dimension
                ]
            )

            if bin_count <= 0:
                valid = False
                break

            if bin_size <= 0:
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
                row_index,
                dimension,
            ] = coordinate

        if not valid:
            coordinates[
                row_index,
                :
            ] = -1

    return coordinates
