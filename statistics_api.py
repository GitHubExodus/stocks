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
            ~np.isnan(values)
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
        q1,
        q3,
        maximum,
        configured_bin_count,
    ):
        """
        Calculate the final bin count and bin size.

        Calculation:

            IQR range = Q3 - Q1

            Full range = max - min

            Initial bin size =
                IQR range / configured bin count

            Difference =
                Full range - IQR range

            Additional bins =
                round(difference / initial bin size)

            Final bin count =
                configured bin count + additional bins

            Final bin size =
                Full range / final bin count
        """

        if configured_bin_count <= 0:
            raise ValueError(
                "configured_bin_count must be greater than 0."
            )

        iqr_range = q3 - q1
        full_range = maximum - minimum

        # No usable range.
        if full_range <= 0:
            return {
                "bin_count": configured_bin_count,
                "bin_size": 0.0,
            }

        # IQR is zero.
        if iqr_range <= 0:
            return {
                "bin_count": configured_bin_count,
                "bin_size": (
                    full_range / configured_bin_count
                ),
            }

        initial_bin_size = (
            iqr_range / configured_bin_count
        )

        difference = (
            full_range - iqr_range
        )

        additional_bins = round(
            difference / initial_bin_size
        )

        final_bin_count = (
            configured_bin_count
            + additional_bins
        )

        final_bin_count = max(
            1,
            final_bin_count,
        )

        final_bin_size = (
            full_range / final_bin_count
        )

        return {
            "bin_count": final_bin_count,
            "bin_size": final_bin_size,
        }