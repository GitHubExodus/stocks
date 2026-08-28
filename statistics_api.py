import numpy as np


class StatisticsAPI:
    """
    Calculates statistics and bin configuration for input values.

    Statistics returned:
        min
        q1
        median
        q3
        max
        sum
        count
    """

    @staticmethod
    def calculate_statistics(input_values):
        """
        Calculate descriptive statistics.

        NaN values are ignored.

        Average is not stored because it can always be calculated:

            average = sum / count
        """

        values = np.asarray(
            input_values,
            dtype=float,
        )

        valid_values = values[
            np.isfinite(values)
        ]

        if len(valid_values) == 0:
            return {
                "min": np.nan,
                "q1": np.nan,
                "median": np.nan,
                "q3": np.nan,
                "max": np.nan,
                "sum": np.nan,
                "count": 0,
            }

        return {
            "min": np.min(valid_values),
            "q1": np.percentile(valid_values, 25),
            "median": np.median(valid_values),
            "q3": np.percentile(valid_values, 75),
            "max": np.max(valid_values),
            "sum": np.sum(valid_values),
            "count": len(valid_values),
        }

    @staticmethod
    def calculate_bin_configuration(
        minimum,
        maximum,
        configured_bin_count,
    ):
        """
        Create a fixed-count bin configuration.

        The configured bin count is always preserved.

        The full range from minimum to maximum is divided
        evenly across the configured number of bins.

        Example:

            minimum = 0
            maximum = 100
            configured_bin_count = 10

            bin_size = 10

            Bins:
                0-10
                10-20
                ...
                90-100
        """

        if configured_bin_count <= 0:
            raise ValueError(
                "configured_bin_count must be greater than 0."
            )

        full_range = maximum - minimum

        if full_range <= 0:
            return {
                "bin_count": configured_bin_count,
                "bin_size": 0.0,
            }

        bin_size = (
            full_range
            / configured_bin_count
        )

        return {
            "bin_count": configured_bin_count,
            "bin_size": bin_size,
        }