import numpy as np
import pandas as pd

from statistics_api import StatisticsAPI


class GridTable:
    """
    Sparse grid table.

    Each input is one dimension of the grid.

    Each populated grid coordinate stores the accumulated
    statistics of every simulation that landed in that cell.
    """

    def __init__(
        self,
        input_names,
        bin_counts,
        bin_sizes,
        mins,
        q1s,
        medians,
        q3s,
        maxs,
    ):
        self.input_names = list(input_names)

        self.bin_counts = np.asarray(
            bin_counts,
            dtype=int,
        )

        self.bin_sizes = np.asarray(
            bin_sizes,
            dtype=float,
        )

        self.mins = np.asarray(
            mins,
            dtype=float,
        )

        self.q1s = np.asarray(
            q1s,
            dtype=float,
        )

        self.medians = np.asarray(
            medians,
            dtype=float,
        )

        self.q3s = np.asarray(
            q3s,
            dtype=float,
        )

        self.maxs = np.asarray(
            maxs,
            dtype=float,
        )

        self.grid = {}

    # ============================================================
    # GRID COORDINATE
    # ============================================================

    def get_grid_coordinate(self, input_values):
        """
        Convert one input row into a grid coordinate.

        Each input is converted independently using:

            bin = floor((value - min) / bin_size)

        The resulting bin is constrained to:

            0 -> bin_count - 1

        NaN values are not allowed to create invalid coordinates.
        """

        values = np.asarray(
            input_values,
            dtype=float,
        )

        if len(values) != len(self.input_names):
            raise ValueError(
                "Number of input values does not match "
                "number of grid dimensions."
            )

        coordinates = []

        for index, value in enumerate(values):

            bin_count = self.bin_counts[index]
            bin_size = self.bin_sizes[index]
            minimum = self.mins[index]

            if np.isnan(value):
                raise ValueError(
                    f"Input '{self.input_names[index]}' "
                    "contains NaN and cannot be assigned "
                    "to a grid coordinate."
                )

            if bin_size <= 0:
                coordinate = 0

            else:
                coordinate = int(
                    np.floor(
                        (value - minimum)
                        / bin_size
                    )
                )

                coordinate = max(
                    0,
                    min(
                        coordinate,
                        bin_count - 1,
                    ),
                )

            coordinates.append(coordinate)

        return tuple(coordinates)

    # ============================================================
    # ADD STATE
    # ============================================================

    def add_state(
        self,
        grid_coordinate,
        statistics,
    ):
        """
        Add one simulation's statistics to a grid cell.

        If the cell does not exist, it is created.

        statistics must contain:

            N
            Sum_R
            Sum_R2
            Sum_D2
            Sum_R3
            Max_Equity
            End_Equity
            Max_DD
            start_timestamp
            end_timestamp
            held_overnight

        The cell also maintains Trade_Count so averages
        can be calculated correctly later.
        """

        coordinate = tuple(
            grid_coordinate
        )

        if coordinate not in self.grid:
            self.grid[coordinate] = {
                "Trade_Count": 0,

                "N": 0,

                "Sum_R": 0.0,
                "Sum_R2": 0.0,
                "Sum_D2": 0.0,
                "Sum_R3": 0.0,

                "Max_Equity": -np.inf,
                "Sum_Max_Equity": 0.0,

                "Sum_End_Equity": 0.0,

                "Max_DD": -np.inf,
                "Sum_Max_DD": 0.0,

                "Start_Timestamp": None,
                "End_Timestamp": None,

                "Held_Overnight": 0,
            }

        cell = self.grid[coordinate]

        # --------------------------------------------------------
        # Count
        # --------------------------------------------------------

        cell["Trade_Count"] += 1

        # --------------------------------------------------------
        # N
        # --------------------------------------------------------

        cell["N"] += statistics["N"]

        # --------------------------------------------------------
        # Return Statistics
        # --------------------------------------------------------

        cell["Sum_R"] += statistics["Sum_R"]

        cell["Sum_R2"] += statistics["Sum_R2"]

        cell["Sum_D2"] += statistics["Sum_D2"]

        cell["Sum_R3"] += statistics["Sum_R3"]

        # --------------------------------------------------------
        # Max Equity
        # --------------------------------------------------------

        max_equity = statistics["Max_Equity"]

        cell["Max_Equity"] = max(
            cell["Max_Equity"],
            max_equity,
        )

        cell["Sum_Max_Equity"] += max_equity

        # --------------------------------------------------------
        # End Equity
        # --------------------------------------------------------

        cell["Sum_End_Equity"] += (
            statistics["End_Equity"]
        )

        # --------------------------------------------------------
        # Maximum Drawdown
        # --------------------------------------------------------

        max_dd = statistics["Max_DD"]

        cell["Max_DD"] = max(
            cell["Max_DD"],
            max_dd,
        )

        cell["Sum_Max_DD"] += max_dd

        # --------------------------------------------------------
        # Timestamps
        # --------------------------------------------------------

        start_timestamp = (
            statistics["start_timestamp"]
        )

        end_timestamp = (
            statistics["end_timestamp"]
        )

        if (
            cell["Start_Timestamp"] is None
            or start_timestamp
            < cell["Start_Timestamp"]
        ):
            cell["Start_Timestamp"] = (
                start_timestamp
            )

        if (
            cell["End_Timestamp"] is None
            or end_timestamp
            > cell["End_Timestamp"]
        ):
            cell["End_Timestamp"] = (
                end_timestamp
            )

        # --------------------------------------------------------
        # Overnight
        # --------------------------------------------------------

        cell["Held_Overnight"] += int(
            statistics["held_overnight"]
        )

    # ============================================================
    # GET GRID DATA
    # ============================================================

    def get_grid_data(self):
        """
        Convert the populated grid into a DataFrame.

        One row represents one populated grid cell.
        """

        rows = []

        for coordinate, cell in self.grid.items():

            trade_count = cell["Trade_Count"]

            row = {}

            # ----------------------------------------------------
            # Grid Coordinates
            # ----------------------------------------------------

            for index, input_name in enumerate(
                self.input_names
            ):
                row[
                    f"dim_{index}"
                ] = coordinate[index]

            # ----------------------------------------------------
            # Accumulated Statistics
            # ----------------------------------------------------

            row["Trade_Count"] = trade_count

            row["N"] = cell["N"]

            row["Avg_N"] = (
                cell["N"] / trade_count
            )

            row["Sum_R"] = cell["Sum_R"]

            row["Avg_R"] = (
                cell["Sum_R"]
                / trade_count
            )

            row["Sum_R2"] = cell["Sum_R2"]

            row["Sum_D2"] = cell["Sum_D2"]

            row["Sum_R3"] = cell["Sum_R3"]

            row["Max_Equity"] = (
                cell["Max_Equity"]
            )

            row["Avg_Max_Equity"] = (
                cell["Sum_Max_Equity"]
                / trade_count
            )

            row["End_Equity"] = (
                cell["Sum_End_Equity"]
            )

            row["Avg_End_Equity"] = (
                cell["Sum_End_Equity"]
                / trade_count
            )

            row["Max_DD"] = cell["Max_DD"]

            row["Avg_Max_DD"] = (
                cell["Sum_Max_DD"]
                / trade_count
            )

            # ----------------------------------------------------
            # Time Information
            # ----------------------------------------------------

            row["Start_Timestamp"] = (
                cell["Start_Timestamp"]
            )

            row["End_Timestamp"] = (
                cell["End_Timestamp"]
            )

            row["Held_Overnight"] = (
                cell["Held_Overnight"]
            )

            rows.append(row)

        return pd.DataFrame(rows)

    # ============================================================
    # GET GRID CONFIGURATION
    # ============================================================

    def get_grid_configuration(self):
        """
        Return the configuration of every grid dimension.

        One row represents one input/dimension.
        """

        rows = []

        for index, input_name in enumerate(
            self.input_names
        ):

            rows.append({
                "dimension_index": index,
                "input_name": input_name,
                "bin_count": self.bin_counts[index],
                "bin_size": self.bin_sizes[index],
                "min": self.mins[index],
                "q1": self.q1s[index],
                "median": self.medians[index],
                "q3": self.q3s[index],
                "max": self.maxs[index],
            })

        return pd.DataFrame(rows)