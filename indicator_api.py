import numpy as np
import pandas as pd

from numba import njit
from zoneinfo import ZoneInfo


class IndicatorAPI:
    """
    Calculates all indicators and returns final input-ready values.

    Numerical indicator calculations use NumPy + Numba.

    Input:
        raw_stock_data:
            DataFrame containing:
                Symbol
                timestamp
                open
                high
                low
                close
                vol
                trade_count
                vwap

    All timestamps are expected to be UTC.

    The Input Stage performs no transformations on the returned values.
    """

    PERIODS = [
        3,
        5,
        9,
        14,
        21,
        50,
        100,
        200,
        500,
        1000,
    ]

    NEW_YORK_TIMEZONE = ZoneInfo(
        "America/New_York"
    )

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self, raw_stock_data):

        self.data = raw_stock_data.copy()

        # ========================================================
        # TIMESTAMP
        # ========================================================

        self.data["timestamp"] = pd.to_datetime(
            self.data["timestamp"],
            utc=True,
        )

        self.data = (
            self.data
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # ========================================================
        # NUMPY PRICE DATA
        # ========================================================

        self.close = (
            self.data["close"]
            .to_numpy(dtype=np.float64)
        )

        self.high = (
            self.data["high"]
            .to_numpy(dtype=np.float64)
        )

        self.low = (
            self.data["low"]
            .to_numpy(dtype=np.float64)
        )

        self.volume = (
            self.data["volume"]
            .to_numpy(dtype=np.float64)
        )

        self.vwap = (
            self.data["vwap"]
            .to_numpy(dtype=np.float64)
        )

    # ============================================================
    # CURRENT TIME
    # ============================================================

    def calculate_current_time(self):
        """
        Returns minutes since 4:00 AM New York time.

        The trading day runs from 4:00 AM through
        3:59 AM the following calendar day.

        UTC timestamps are converted to
        America/New_York first.

        ZoneInfo handles EST/EDT automatically.
        """

        local_time = (
            self.data["timestamp"]
            .dt.tz_convert(
                self.NEW_YORK_TIMEZONE
            )
        )

        session_date = (
            local_time.dt.normalize()
            - pd.to_timedelta(
                (
                    local_time.dt.hour < 4
                ).astype(int),
                unit="D",
            )
        )

        start_of_session = (
            session_date
            + pd.Timedelta(hours=4)
        )

        minutes = (
            local_time
            - start_of_session
        ).dt.total_seconds() / 60.0

        return minutes

    # ============================================================
    # DOLLAR VOLUME
    # ============================================================

    def calculate_dollar_volume(self, period):
        """
        Calculates dollar volume.

        Per bar:
            dollar_volume = volume × close

        If period > 1:
            returns the sum of the per-bar dollar volumes
            over the rolling period.
        """

        dollar_volume = (
            self.volume
            * self.close
        )

        result = _rolling_sum(
            dollar_volume,
            period,
        )

        return pd.Series(
            result,
            index=self.data.index,
        )

    # ============================================================
    # EMA DISTANCE
    # ============================================================

    def calculate_ema_distance(self, period):
        """
        Percentage distance between current close and EMA.

            (close - EMA) / EMA * 100
        """

        ema = _ema(
            self.close,
            period,
        )

        result = (
            (
                self.close
                - ema
            )
            / ema
        ) * 100.0

        return pd.Series(
            result,
            index=self.data.index,
        )

    # ============================================================
    # DEMA DISTANCE
    # ============================================================

    def calculate_dema_distance(self, period):
        """
        Double Exponential Moving Average.

            DEMA = 2 * EMA1 - EMA2

        Returns percentage distance between close and DEMA.
        """

        ema1 = _ema(
            self.close,
            period,
        )

        ema2 = _ema(
            ema1,
            period,
        )

        dema = (
            2.0 * ema1
            - ema2
        )

        result = (
            (
                self.close
                - dema
            )
            / dema
        ) * 100.0

        return pd.Series(
            result,
            index=self.data.index,
        )

    # ============================================================
    # VWAP DISTANCE
    # ============================================================

    def calculate_vwap_distance(self):
        """
        Percentage distance between current close and VWAP.

            (close - VWAP) / VWAP * 100
        """

        result = (
            (
                self.close
                - self.vwap
            )
            / self.vwap
        ) * 100.0

        return pd.Series(
            result,
            index=self.data.index,
        )

    # ============================================================
    # RSI
    # ============================================================

    def calculate_rsi(self, period):
        """
        Wilder-style RSI.

        Returns values from 0 through 100.
        """

        delta = np.empty(
            len(self.close),
            dtype=np.float64,
        )

        delta[:] = np.nan

        for i in range(1, len(self.close)):

            if (
                np.isfinite(self.close[i])
                and np.isfinite(self.close[i - 1])
            ):
                delta[i] = (
                    self.close[i]
                    - self.close[i - 1]
                )

        gains = np.zeros(
            len(delta),
            dtype=np.float64,
        )

        losses = np.zeros(
            len(delta),
            dtype=np.float64,
        )

        for i in range(len(delta)):

            if not np.isfinite(delta[i]):
                gains[i] = np.nan
                losses[i] = np.nan

            elif delta[i] > 0:
                gains[i] = delta[i]
                losses[i] = 0.0

            elif delta[i] < 0:
                gains[i] = 0.0
                losses[i] = -delta[i]

            else:
                gains[i] = 0.0
                losses[i] = 0.0

        average_gain = _ema(
            gains,
            period,
        )

        average_loss = _ema(
            losses,
            period,
        )

        rsi = np.full(
            len(self.close),
            np.nan,
            dtype=np.float64,
        )

        for i in range(len(rsi)):

            gain = average_gain[i]
            loss = average_loss[i]

            if (
                not np.isfinite(gain)
                or not np.isfinite(loss)
            ):
                continue

            # No losses means RSI = 100.
            if loss == 0.0:

                # No gains and no losses.
                if gain == 0.0:
                    rsi[i] = np.nan

                else:
                    rsi[i] = 100.0

                continue

            rs = gain / loss

            rsi[i] = (
                100.0
                - (
                    100.0
                    / (1.0 + rs)
                )
            )

        return pd.Series(
            rsi,
            index=self.data.index,
        )

    # ============================================================
    # ROC
    # ============================================================

    def calculate_roc(self, period):
        """
        Rate of Change as a percentage.
        """

        result = np.full(
            len(self.close),
            np.nan,
            dtype=np.float64,
        )

        for i in range(period, len(self.close)):

            current = self.close[i]
            previous = self.close[
                i - period
            ]

            if (
                np.isfinite(current)
                and np.isfinite(previous)
                and previous != 0.0
            ):
                result[i] = (
                    (
                        current
                        / previous
                    )
                    - 1.0
                ) * 100.0

        return pd.Series(
            result,
            index=self.data.index,
        )

    # ============================================================
    # RETURN STANDARD DEVIATION
    # ============================================================

    def calculate_return_standard_deviation(
        self,
        period,
    ):
        """
        Standard deviation of percentage returns.

        Uses sample standard deviation,
        matching pandas rolling().std() default ddof=1.
        """

        returns = np.full(
            len(self.close),
            np.nan,
            dtype=np.float64,
        )

        for i in range(1, len(self.close)):

            current = self.close[i]
            previous = self.close[
                i - 1
            ]

            if (
                np.isfinite(current)
                and np.isfinite(previous)
                and previous != 0.0
            ):
                returns[i] = (
                    (
                        current
                        / previous
                    )
                    - 1.0
                ) * 100.0

        result = _rolling_std(
            returns,
            period,
        )

        return pd.Series(
            result,
            index=self.data.index,
        )

    # ============================================================
    # ATR
    # ============================================================

    def calculate_normalized_atr(self, period):
        """
        Calculates ATR and normalizes it by current close.

            ATR / current close * 100
        """

        true_range = _calculate_true_range(
            self.high,
            self.low,
            self.close,
        )

        atr = _ema(
            true_range,
            period,
        )

        result = (
            atr
            / self.close
        ) * 100.0

        return pd.Series(
            result,
            index=self.data.index,
        )

    # ============================================================
    # DX
    # ============================================================

    def calculate_dx(self, period):
        """
        Calculates Directional Index (DX).

        Implemented with NumPy + Numba.
        """

        dx = _calculate_dx(
            self.high,
            self.low,
            self.close,
            period,
        )

        return pd.Series(
            dx,
            index=self.data.index,
        )

    # ============================================================
    # PIVOT HIGH
    # ============================================================

    def calculate_pivot_high(self, period):
        """
        Returns:

            pivot_high_distance
            pivot_high_minutes_away

        A pivot high requires:

            period bars to the left
            period bars to the right

        The pivot becomes available only after the
        look-right period has passed.

        The selected pivot is the newest fully calculated
        pivot high.
        """

        pivot_values, pivot_indices = (
            _calculate_pivots(
                self.high,
                period,
                True,
            )
        )

        distance = np.full(
            len(self.close),
            np.nan,
            dtype=np.float64,
        )

        for i in range(len(self.close)):

            pivot_value = pivot_values[i]

            if (
                np.isfinite(pivot_value)
                and pivot_value != 0.0
                and np.isfinite(self.close[i])
            ):
                distance[i] = (
                    (
                        self.close[i]
                        - pivot_value
                    )
                    / pivot_value
                ) * 100.0

        timestamps = (
            self.data["timestamp"]
        )

        pivot_available_timestamp = (
            pd.Series(
                pd.NaT,
                index=self.data.index,
                dtype="datetime64[ns, UTC]",
            )
        )

        valid_indices = (
            pivot_indices >= 0
        )

        pivot_available_timestamp.loc[
            valid_indices
        ] = timestamps.iloc[
            pivot_indices[
                valid_indices
            ]
        ].to_numpy()

        minutes_away = (
            timestamps
            - pivot_available_timestamp
        ).dt.total_seconds() / 60.0

        return pd.DataFrame(
            {
                "pivot_high_distance":
                    distance,

                "pivot_high_minutes_away":
                    minutes_away,
            },
            index=self.data.index,
        )

    # ============================================================
    # PIVOT LOW
    # ============================================================

    def calculate_pivot_low(self, period):
        """
        Returns:

            pivot_low_distance
            pivot_low_minutes_away
        """

        pivot_values, pivot_indices = (
            _calculate_pivots(
                self.low,
                period,
                False,
            )
        )

        distance = np.full(
            len(self.close),
            np.nan,
            dtype=np.float64,
        )

        for i in range(len(self.close)):

            pivot_value = pivot_values[i]

            if (
                np.isfinite(pivot_value)
                and pivot_value != 0.0
                and np.isfinite(self.close[i])
            ):
                distance[i] = (
                    (
                        self.close[i]
                        - pivot_value
                    )
                    / pivot_value
                ) * 100.0

        timestamps = (
            self.data["timestamp"]
        )

        pivot_available_timestamp = (
            pd.Series(
                pd.NaT,
                index=self.data.index,
                dtype="datetime64[ns, UTC]",
            )
        )

        valid_indices = (
            pivot_indices >= 0
        )

        pivot_available_timestamp.loc[
            valid_indices
        ] = timestamps.iloc[
            pivot_indices[
                valid_indices
            ]
        ].to_numpy()

        minutes_away = (
            timestamps
            - pivot_available_timestamp
        ).dt.total_seconds() / 60.0

        return pd.DataFrame(
            {
                "pivot_low_distance":
                    distance,

                "pivot_low_minutes_away":
                    minutes_away,
            },
            index=self.data.index,
        )

    # ============================================================
    # PIVOT INTERNAL CALCULATION
    # ============================================================

    def _calculate_pivots(
        self,
        values,
        period,
        mode,
    ):
        """
        Compatibility wrapper around the Numba pivot calculation.
        """

        values_array = np.asarray(
            values,
            dtype=np.float64,
        )

        if mode == "high":

            pivot_values, pivot_indices = (
                _calculate_pivots(
                    values_array,
                    period,
                    True,
                )
            )

        elif mode == "low":

            pivot_values, pivot_indices = (
                _calculate_pivots(
                    values_array,
                    period,
                    False,
                )
            )

        else:

            raise ValueError(
                "mode must be 'high' or 'low'"
            )

        timestamps = self.data[
            "timestamp"
        ]

        pivot_available_times = pd.Series(
            pd.NaT,
            index=self.data.index,
            dtype="datetime64[ns, UTC]",
        )

        valid_indices = (
            pivot_indices >= 0
        )

        pivot_available_times.loc[
            valid_indices
        ] = timestamps.iloc[
            pivot_indices[
                valid_indices
            ]
        ].to_numpy()

        return pd.DataFrame(
            {
                "value":
                    pivot_values,

                "available_timestamp":
                    pivot_available_times,
            },
            index=self.data.index,
        )


# =================================================================
# NUMPY / NUMBA FUNCTIONS
# =================================================================


@njit
def _rolling_sum(
    values,
    period,
):
    """
    Rolling sum.

    Equivalent to:

        pandas.Series(values).rolling(
            window=period,
            min_periods=period,
        ).sum()
    """

    n = len(values)

    result = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    if period <= 0:
        return result

    running_sum = 0.0
    valid_count = 0

    for i in range(n):

        current = values[i]

        if np.isfinite(current):

            running_sum += current
            valid_count += 1

        if i >= period:

            old = values[
                i - period
            ]

            if np.isfinite(old):

                running_sum -= old
                valid_count -= 1

        if (
            i >= period - 1
            and valid_count == period
        ):

            result[i] = running_sum

    return result


@njit
def _ema(
    values,
    period,
):
    """
    EMA using:

        alpha = 1 / period

    and adjust=False behavior.

    Output remains NaN until `period` valid
    observations have been received.
    """

    n = len(values)

    result = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    if period <= 0:
        return result

    alpha = 1.0 / period

    initialized = False
    ema_value = np.nan
    valid_count = 0

    for i in range(n):

        value = values[i]

        if not np.isfinite(value):
            continue

        if not initialized:

            ema_value = value
            initialized = True

        else:

            ema_value = (
                alpha * value
                + (1.0 - alpha)
                * ema_value
            )

        valid_count += 1

        if valid_count >= period:

            result[i] = ema_value

    return result


@njit
def _rolling_std(
    values,
    period,
):
    """
    Rolling sample standard deviation.

    Uses ddof=1, matching pandas:

        rolling().std()
    """

    n = len(values)

    result = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    if period <= 0:
        return result

    for i in range(
        period - 1,
        n,
    ):

        start = (
            i - period + 1
        )

        count = 0
        total = 0.0

        for j in range(
            start,
            i + 1,
        ):

            value = values[j]

            if np.isfinite(value):

                total += value
                count += 1

        if count < period:
            continue

        mean = (
            total / count
        )

        squared_sum = 0.0

        for j in range(
            start,
            i + 1,
        ):

            value = values[j]

            if np.isfinite(value):

                difference = (
                    value - mean
                )

                squared_sum += (
                    difference
                    * difference
                )

        if count > 1:

            variance = (
                squared_sum
                / (count - 1)
            )

            result[i] = np.sqrt(
                variance
            )

    return result


@njit
def _calculate_true_range(
    high,
    low,
    close,
):
    """
    Calculate True Range.

    Equivalent to the three-way maximum used
    by the original pandas implementation.
    """

    n = len(close)

    result = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    for i in range(n):

        current_high = high[i]
        current_low = low[i]

        if (
            not np.isfinite(current_high)
            or not np.isfinite(current_low)
        ):
            continue

        # First bar:
        #
        # Previous close is unavailable, so
        # high - low is used.

        if i == 0:

            result[i] = (
                current_high
                - current_low
            )

            continue

        previous_close = close[
            i - 1
        ]

        if not np.isfinite(
            previous_close
        ):

            result[i] = (
                current_high
                - current_low
            )

            continue

        range_1 = (
            current_high
            - current_low
        )

        range_2 = abs(
            current_high
            - previous_close
        )

        range_3 = abs(
            current_low
            - previous_close
        )

        maximum = range_1

        if range_2 > maximum:
            maximum = range_2

        if range_3 > maximum:
            maximum = range_3

        result[i] = maximum

    return result


@njit
def _calculate_dx(
    high,
    low,
    close,
    period,
):
    """
    Calculate Directional Index (DX).

    Uses Wilder-style smoothing:

        alpha = 1 / period
    """

    n = len(close)

    previous_high = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    previous_low = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    previous_close = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    for i in range(1, n):

        previous_high[i] = high[
            i - 1
        ]

        previous_low[i] = low[
            i - 1
        ]

        previous_close[i] = close[
            i - 1
        ]

    up_move = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    down_move = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    plus_dm = np.zeros(
        n,
        dtype=np.float64,
    )

    minus_dm = np.zeros(
        n,
        dtype=np.float64,
    )

    for i in range(1, n):

        if (
            not np.isfinite(high[i])
            or not np.isfinite(low[i])
            or not np.isfinite(
                previous_high[i]
            )
            or not np.isfinite(
                previous_low[i]
            )
        ):
            continue

        up_move[i] = (
            high[i]
            - previous_high[i]
        )

        down_move[i] = (
            previous_low[i]
            - low[i]
        )

        if (
            up_move[i]
            > down_move[i]
            and up_move[i] > 0.0
        ):

            plus_dm[i] = up_move[i]

        else:

            plus_dm[i] = 0.0

        if (
            down_move[i]
            > up_move[i]
            and down_move[i] > 0.0
        ):

            minus_dm[i] = down_move[i]

        else:

            minus_dm[i] = 0.0

    true_range = _calculate_true_range(
        high,
        low,
        close,
    )

    atr = _ema(
        true_range,
        period,
    )

    plus_dm_smoothed = _ema(
        plus_dm,
        period,
    )

    minus_dm_smoothed = _ema(
        minus_dm,
        period,
    )

    dx = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    for i in range(n):

        if (
            not np.isfinite(atr[i])
            or not np.isfinite(
                plus_dm_smoothed[i]
            )
            or not np.isfinite(
                minus_dm_smoothed[i]
            )
        ):
            continue

        if atr[i] == 0.0:

            plus_di = 0.0
            minus_di = 0.0

        else:

            plus_di = (
                100.0
                * plus_dm_smoothed[i]
                / atr[i]
            )

            minus_di = (
                100.0
                * minus_dm_smoothed[i]
                / atr[i]
            )

        denominator = (
            plus_di
            + minus_di
        )

        if denominator == 0.0:

            dx[i] = 0.0

        else:

            dx[i] = (
                100.0
                * abs(
                    plus_di
                    - minus_di
                )
                / denominator
            )

    return dx


@njit
def _calculate_pivots(
    values,
    period,
    is_high,
):
    """
    Calculate fully confirmed pivots.

    A candidate at index i becomes available at:

        i + period

    The newest confirmed pivot is carried forward.

    is_high=True:
        pivot high

    is_high=False:
        pivot low
    """

    n = len(values)

    pivot_values = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    pivot_indices = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    latest_pivot_value = np.nan
    latest_pivot_index = -1

    if period <= 0:
        return (
            pivot_values,
            pivot_indices,
        )

    for current_index in range(n):

        candidate_index = (
            current_index - period
        )

        # --------------------------------------------------------
        # Not enough bars to have period bars
        # on both sides.
        # --------------------------------------------------------

        if candidate_index < period:

            pivot_values[
                current_index
            ] = latest_pivot_value

            pivot_indices[
                current_index
            ] = latest_pivot_index

            continue

        left_start = (
            candidate_index
            - period
        )

        left_end = candidate_index

        right_start = (
            candidate_index + 1
        )

        right_end = (
            candidate_index
            + period
            + 1
        )

        candidate_value = values[
            candidate_index
        ]

        valid = np.isfinite(
            candidate_value
        )

        # --------------------------------------------------------
        # Check left side.
        # --------------------------------------------------------

        if valid:

            for j in range(
                left_start,
                left_end,
            ):

                if not np.isfinite(
                    values[j]
                ):

                    valid = False
                    break

        # --------------------------------------------------------
        # Check right side.
        # --------------------------------------------------------

        if valid:

            for j in range(
                right_start,
                right_end,
            ):

                if not np.isfinite(
                    values[j]
                ):

                    valid = False
                    break

        # --------------------------------------------------------
        # Determine whether candidate is a pivot.
        # --------------------------------------------------------

        if valid:

            is_pivot = True

            if is_high:

                for j in range(
                    left_start,
                    left_end,
                ):

                    if (
                        candidate_value
                        <= values[j]
                    ):

                        is_pivot = False
                        break

                if is_pivot:

                    for j in range(
                        right_start,
                        right_end,
                    ):

                        if (
                            candidate_value
                            <= values[j]
                        ):

                            is_pivot = False
                            break

            else:

                for j in range(
                    left_start,
                    left_end,
                ):

                    if (
                        candidate_value
                        >= values[j]
                    ):

                        is_pivot = False
                        break

                if is_pivot:

                    for j in range(
                        right_start,
                        right_end,
                    ):

                        if (
                            candidate_value
                            >= values[j]
                        ):

                            is_pivot = False
                            break

            if is_pivot:

                latest_pivot_value = (
                    candidate_value
                )

                # Important:
                #
                # Store the candidate's index.
                #
                # The pivot is confirmed at current_index,
                # but the pivot itself occurred at candidate_index.

                latest_pivot_index = (
                    candidate_index
                )

        # --------------------------------------------------------
        # Carry latest confirmed pivot forward.
        # --------------------------------------------------------

        pivot_values[
            current_index
        ] = latest_pivot_value

        pivot_indices[
            current_index
        ] = latest_pivot_index

    return (
        pivot_values,
        pivot_indices,
    )
