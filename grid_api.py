import numpy as np
import pandas as pd


class GridTable:
    """
    Sparse grid table.

    Each input is one dimension of the grid.

    Each populated grid coordinate stores accumulated
    trade statistics for every RR simulation that
    landed in that cell.
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
            dtype=np.int64,
        )

        self.bin_sizes = np.asarray(
            bin_sizes,
            dtype=np.float64,
        )

        self.mins = np.asarray(
            mins,
            dtype=np.float64,
        )

        self.q1s = np.asarray(
            q1s,
            dtype=np.float64,
        )

        self.medians = np.asarray(
            medians,
            dtype=np.float64,
        )

        self.q3s = np.asarray(
            q3s,
            dtype=np.float64,
        )

        self.maxs = np.asarray(
            maxs,
            dtype=np.float64,
        )

        self.grid = {}

    # ============================================================
    # GRID COORDINATE
    # ============================================================

    def get_grid_coordinate(
        self,
        input_values,
    ):
        """
        Convert one input row into a grid coordinate.

        Each input is converted independently using:

            bin = floor((value - min) / bin_size)

        The resulting bin is constrained to:

            0 -> bin_count - 1
        """

        values = np.asarray(
            input_values,
            dtype=np.float64,
        )

        if len(values) != len(
            self.input_names
        ):
            raise ValueError(
                "Number of input values does not match "
                "number of grid dimensions."
            )

        coordinates = []

        for index, value in enumerate(values):

            if not np.isfinite(value):
                raise ValueError(
                    f"Input '{self.input_names[index]}' "
                    "contains an invalid value."
                )

            bin_count = self.bin_counts[index]
            bin_size = self.bin_sizes[index]
            minimum = self.mins[index]

            if bin_count <= 0:
                raise ValueError(
                    f"Invalid bin count for "
                    f"'{self.input_names[index]}'."
                )

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

                coordinate = max(
                    0,
                    min(
                        coordinate,
                        bin_count - 1,
                    ),
                )

            coordinates.append(
                coordinate
            )

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
        Add one RR trade to a grid cell.

        Accumulated values:

            Total_Trades
            Total_Time_Elapsed_Minutes
            Total_Return_Percent
            Sum_Return_Squared
        """

        coordinate = tuple(
            grid_coordinate
        )

        if coordinate not in self.grid:

            self.grid[coordinate] = {
                "Total_Trades": 0,

                "Total_Wins": 0,

                "Total_Losses": 0,

                "Total_Win_Return_Percent": 0.0,

                "Total_Loss_Return_Percent": 0.0,

                "Total_Time_Elapsed_Minutes": 0.0,

                "Total_Return_Percent": 0.0,

                "Sum_Return_Squared": 0.0,
            }

        cell = self.grid[
            coordinate
        ]

        time_elapsed = float(
            statistics[
                "time_elapsed_minutes"
            ]
        )

        trade_return = float(
            statistics[
                "trade_return_percent"
            ]
        )


        # --------------------------------------------------------
        # Total trades
        # --------------------------------------------------------

        cell[
            "Total_Trades"
        ] += 1


        # --------------------------------------------------------
        # Wins and losses
        # --------------------------------------------------------

        if trade_return > 0:

            cell[
                "Total_Wins"
            ] += 1

            cell[
                "Total_Win_Return_Percent"
            ] += trade_return

        elif trade_return <= 0:

            cell[
                "Total_Losses"
            ] += 1

            cell[
                "Total_Loss_Return_Percent"
            ] += trade_return


        # --------------------------------------------------------
        # Total elapsed time
        # --------------------------------------------------------

        cell[
            "Total_Time_Elapsed_Minutes"
        ] += time_elapsed

        # --------------------------------------------------------
        # Total return
        # --------------------------------------------------------

        cell[
            "Total_Return_Percent"
        ] += trade_return

        # --------------------------------------------------------
        # Sum of squared returns
        # --------------------------------------------------------

        cell[
            "Sum_Return_Squared"
        ] += (
            trade_return
            * trade_return
        )

    # ============================================================
    # GET GRID DATA
    # ============================================================

    def get_grid_data(self):
        """
        Convert the populated grid into a DataFrame.

        One row represents one populated grid cell.

        Totals are stored so averages can be calculated
        without storing every individual trade.
        """

        rows = []

        for coordinate, cell in (
            self.grid.items()
        ):

            total_trades = (
                cell["Total_Trades"]
            )

            row = {}

            # ----------------------------------------------------
            # Grid coordinates
            # ----------------------------------------------------

            for index, input_name in enumerate(
                self.input_names
            ):
                row[
                    f"dim_{index}"
                ] = coordinate[index]

            # ----------------------------------------------------
            # Trade count
            # ----------------------------------------------------

            row[
                "Total_Trades"
            ] = total_trades

            row[
                "Total_Wins"
            ] = cell[
                "Total_Wins"
            ]

            row[
                "Total_Losses"
            ] = cell[
                "Total_Losses"
            ]

            row[
                "Total_Win_Return_Percent"
            ] = cell[
                "Total_Win_Return_Percent"
            ]

            row[
                "Total_Loss_Return_Percent"
            ] = cell[
                "Total_Loss_Return_Percent"
            ]

            # ----------------------------------------------------
            # Time
            # ----------------------------------------------------

            row[
                "Total_Time_Elapsed_Minutes"
            ] = (
                cell[
                    "Total_Time_Elapsed_Minutes"
                ]
            )

            row[
                "Avg_Time_Elapsed_Minutes"
            ] = (
                cell[
                    "Total_Time_Elapsed_Minutes"
                ]
                / total_trades
            )

            # ----------------------------------------------------
            # Return
            # ----------------------------------------------------

            row[
                "Total_Return_Percent"
            ] = (
                cell[
                    "Total_Return_Percent"
                ]
            )

            row[
                "Avg_Return_Percent"
            ] = (
                cell[
                    "Total_Return_Percent"
                ]
                / total_trades
            )

            # ----------------------------------------------------
            # Squared return
            # ----------------------------------------------------

            row[
                "Sum_Return_Squared"
            ] = (
                cell[
                    "Sum_Return_Squared"
                ]
            )

            rows.append(row)

        return pd.DataFrame(
            rows
        )

    # ============================================================
    # GET GRID CONFIGURATION
    # ============================================================

    def get_grid_configuration(
        self,
    ):
        """
        Return the configuration of every
        grid dimension.

        One row represents one input/dimension.
        """

        rows = []

        for index, input_name in enumerate(
            self.input_names
        ):

            rows.append({
                "dimension_index": index,

                "input_name": input_name,

                "bin_count": (
                    self.bin_counts[index]
                ),

                "bin_size": (
                    self.bin_sizes[index]
                ),

                "min": (
                    self.mins[index]
                ),

                "q1": (
                    self.q1s[index]
                ),

                "median": (
                    self.medians[index]
                ),

                "q3": (
                    self.q3s[index]
                ),

                "max": (
                    self.maxs[index]
                ),
            })

        return pd.DataFrame(
            rows
        )