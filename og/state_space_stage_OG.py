import pandas as pd

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


    def _add_row_to_grid(
        self,
        grid_table,
        rr_row,
        input_timestamp_lookup,
        input_names,
    ):
        start_timestamp = pd.Timestamp(
            rr_row["start_timestamp"]
        )

        if (
            start_timestamp
            not in input_timestamp_lookup.index
        ):
            return

        input_row = (
            input_timestamp_lookup.loc[
                start_timestamp
            ]
        )

        # If duplicate timestamps exist,
        # use the first matching row.
        if isinstance(
            input_row,
            pd.DataFrame,
        ):
            input_row = input_row.iloc[0]

        input_values = (
            input_row[
                input_names
            ].to_numpy()
        )

        # Skip rows containing NaN values.
        if pd.isna(input_values).any():
            return

        grid_coordinate = (
            grid_table.get_grid_coordinate(
                input_values
            )
        )

        statistics = {
            "N": rr_row["N"],
            "Sum_R": rr_row["Sum_R"],
            "Sum_R2": rr_row["Sum_R2"],
            "Sum_D2": rr_row["Sum_D2"],
            "Sum_R3": rr_row["Sum_R3"],
            "Max_Equity": rr_row[
                "Max_Equity"
            ],
            "End_Equity": rr_row[
                "End_Equity"
            ],
            "Max_DD": rr_row["Max_DD"],
            "start_timestamp": rr_row[
                "start_timestamp"
            ],
            "end_timestamp": rr_row[
                "end_timestamp"
            ],
            "held_overnight": rr_row[
                "held_overnight"
            ],
        }

        grid_table.add_state(
            grid_coordinate,
            statistics,
        )


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
        self.input_statistics = input_statistics.copy()
        validation_start = pd.Timestamp(
            "2023-01-01",
            tz="UTC",
        )

        # ========================================================
        # 2. VALIDATE BIN COUNTS
        # ========================================================

        input_names = (
            self.input_statistics["input"]
            .tolist()
        )

        missing_bin_counts = [
            input_name
            for input_name in input_names
            if input_name not in bin_counts
        ]

        if missing_bin_counts:
            raise KeyError(
                "Missing bin counts for inputs: "
                + ", ".join(missing_bin_counts)
            )

        # ========================================================
        # 3. CALCULATE BIN CONFIGURATION
        # ========================================================

        bin_counts_final = []
        bin_sizes_final = []

        for index, row in self.input_statistics.iterrows():

            configured_bin_count = bin_counts[
                row["input"]
            ]

            result = (
                StatisticsAPI.calculate_bin_configuration(
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
        # 5. CREATE GRID TABLE
        # ========================================================

        mins = (
            self.input_statistics["min"]
            .tolist()
        )

        q1s = (
            self.input_statistics["q1"]
            .tolist()
        )

        medians = (
            self.input_statistics["median"]
            .tolist()
        )

        q3s = (
            self.input_statistics["q3"]
            .tolist()
        )

        maxs = (
            self.input_statistics["max"]
            .tolist()
        )

        

        # ========================================================
        # 6. COMPUTE TRAINING AND VALIDATION GRIDS
        # ========================================================

        validation_start = pd.Timestamp(
            "2023-01-01",
            tz="UTC",
        )

        # --------------------------------------------------------
        # Split risk/reward data by entry timestamp.
        #
        # A trade belongs to training or validation based on
        # when the trade started, not when it ended.
        # --------------------------------------------------------

        risk_reward_timestamps = pd.to_datetime(
            risk_reward_data["start_timestamp"],
            utc=True,
        )

        training_mask = (
            risk_reward_timestamps
            < validation_start
        )

        validation_mask = (
            risk_reward_timestamps
            >= validation_start
        )

        # --------------------------------------------------------
        # Create a timestamp lookup for input data.
        # --------------------------------------------------------

        input_timestamp_lookup = (
            self.input_data
            .set_index("timestamp")
        )

        # --------------------------------------------------------
        # Create training grid only if training data exists.
        # --------------------------------------------------------

        training_grid_table = None

        if training_mask.any():

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

        # --------------------------------------------------------
        # Create validation grid.
        #
        # It uses the exact same configuration as training.
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Populate training grid.
        # --------------------------------------------------------

        if training_grid_table is not None:

            training_data = (
                risk_reward_data[
                    training_mask
                ]
            )

            for _, rr_row in (
                training_data.iterrows()
            ):

                self._add_row_to_grid(
                    grid_table=training_grid_table,
                    rr_row=rr_row,
                    input_timestamp_lookup=(
                        input_timestamp_lookup
                    ),
                    input_names=input_names,
                )

        # --------------------------------------------------------
        # Populate validation grid.
        # --------------------------------------------------------

        validation_data = (
            risk_reward_data[
                validation_mask
            ]
        )

        for _, rr_row in (
            validation_data.iterrows()
        ):

            self._add_row_to_grid(
                grid_table=validation_grid_table,
                rr_row=rr_row,
                input_timestamp_lookup=(
                    input_timestamp_lookup
                ),
                input_names=input_names,
            )

        # ========================================================
        # 7. GET GRID DATA
        # ========================================================

        training_grid_configuration = None
        training_grid_data = None

        if training_grid_table is not None:

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
        # 8. SAVE GRID
        # ========================================================

        # --------------------------------------------------------
        # Save training grid only if training data exists.
        # --------------------------------------------------------

        if training_grid_table is not None:

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

        # --------------------------------------------------------
        # Save validation grid.
        # --------------------------------------------------------

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

        training_cells = 0

        if training_grid_data is not None:
            training_cells = len(
                training_grid_data
            )

        validation_cells = len(
            validation_grid_data
        )

        log(
            f"State Space Stage completed | "
            f"symbol={symbol} | "
            f"trade_type={trade_type} | "
            f"SL={stop_loss_percentage} | "
            f"RR={risk_reward_ratio} | "
            f"training_cells={training_cells} | "
            f"validation_cells={validation_cells}"
        )

        # ========================================================
        # 9. RETURN
        # ========================================================

        return {
            "training": {
                "grid_configuration": (
                    training_grid_configuration
                ),
                "grid_data": (
                    training_grid_data
                ),
            },

            "validation": {
                "grid_configuration": (
                    validation_grid_configuration
                ),
                "grid_data": (
                    validation_grid_data
                ),
            },

            "input_statistics": (
                self.input_statistics
            ),
        }
        
