import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


class IndicatorAPI:
    """
    Calculates all indicators and returns final input-ready values.

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

    PERIODS = [3, 5, 9, 14, 21, 50, 100, 200, 500, 1000]

    NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")

    def __init__(self, raw_stock_data):

        self.data = raw_stock_data.copy()

        # ========================================================
        # TIMESTAMP
        # ========================================================

        self.data["timestamp"] = pd.to_datetime(
            self.data["timestamp"],
            utc=True,
        )

        # All calculations must happen in chronological order.
        self.data = self.data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # ========================================================
        # PRICE DATA
        # ========================================================

        self.close = self.data["close"].astype(float)
        self.high = self.data["high"].astype(float)
        self.low = self.data["low"].astype(float)
        self.volume = self.data["volume"].astype(float)
        self.vwap = self.data["vwap"].astype(float)

    # ============================================================
    # CURRENT TIME
    # ============================================================

    def calculate_current_time(self):
        """
        Returns minutes since 4:00 AM New York time.

        UTC timestamps are converted to America/New_York first.

        ZoneInfo automatically handles EST/EDT for every date.
        """

        local_time = self.data["timestamp"].dt.tz_convert(
            self.NEW_YORK_TIMEZONE
        )

        start_of_day = (
            local_time.dt.normalize()
            + pd.Timedelta(hours=4)
        )

        minutes = (
            local_time - start_of_day
        ).dt.total_seconds() / 60.0

        return minutes

    # ============================================================
    # DOLLAR VOLUME
    # ============================================================

    def calculate_dollar_volume(self, period):
        """
        Sum volume over the period and multiply by the
        current bar's closing price.
        """

        volume_sum = self.volume.rolling(
            window=period,
            min_periods=period,
        ).sum()

        return volume_sum * self.close

    # ============================================================
    # EMA DISTANCE
    # ============================================================

    def calculate_ema_distance(self, period):
        """
        Percentage distance between current close and EMA.

        (close - EMA) / EMA * 100
        """

        ema = self.close.ewm(
            span=period,
            adjust=False,
            min_periods=period,
        ).mean()

        return (
            (self.close - ema)
            / ema
        ) * 100.0

    # ============================================================
    # DEMA DISTANCE
    # ============================================================

    def calculate_dema_distance(self, period):
        """
        Double Exponential Moving Average.

        DEMA = 2 * EMA1 - EMA2

        Returns percentage distance between close and DEMA.
        """

        ema1 = self.close.ewm(
            span=period,
            adjust=False,
            min_periods=period,
        ).mean()

        ema2 = ema1.ewm(
            span=period,
            adjust=False,
            min_periods=period,
        ).mean()

        dema = (
            2.0 * ema1
            - ema2
        )

        return (
            (self.close - dema)
            / dema
        ) * 100.0

    # ============================================================
    # VWAP DISTANCE
    # ============================================================

    def calculate_vwap_distance(self):
        """
        Percentage distance between current close and VWAP.

        VWAP is already supplied by the raw stock data.
        """

        return (
            (self.close - self.vwap)
            / self.vwap
        ) * 100.0

    # ============================================================
    # RSI
    # ============================================================

    def calculate_rsi(self, period):
        """
        Wilder-style RSI.

        Returns values from 0 through 100.
        """

        delta = self.close.diff()

        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        average_gain = gains.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()

        average_loss = losses.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()

        rs = average_gain / average_loss

        rsi = (
            100.0
            - (
                100.0
                / (1.0 + rs)
            )
        )

        # No losses means RSI is 100.
        rsi = rsi.where(
            average_loss != 0,
            100.0,
        )

        # No gains and no losses.
        rsi = rsi.where(
            ~(
                (average_gain == 0)
                & (average_loss == 0)
            ),
            np.nan,
        )

        return rsi

    # ============================================================
    # ROC
    # ============================================================

    def calculate_roc(self, period):
        """
        Rate of Change as a percentage.
        """

        return (
            (
                self.close
                / self.close.shift(period)
            )
            - 1.0
        ) * 100.0

    # ============================================================
    # RETURN STANDARD DEVIATION
    # ============================================================

    def calculate_return_standard_deviation(self, period):
        """
        Standard deviation of percentage returns.
        """

        returns = (
            self.close.pct_change()
            * 100.0
        )

        return returns.rolling(
            window=period,
            min_periods=period,
        ).std()

    # ============================================================
    # ATR
    # ============================================================

    def calculate_normalized_atr(self, period):
        """
        Calculates ATR and normalizes it by current close.

        Result:

            ATR / current close * 100
        """

        previous_close = self.close.shift(1)

        true_range = pd.concat(
            [
                self.high - self.low,
                (
                    self.high
                    - previous_close
                ).abs(),
                (
                    self.low
                    - previous_close
                ).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()

        return (
            atr
            / self.close
        ) * 100.0

    # ============================================================
    # DX
    # ============================================================

    def calculate_dx(self, period):
        """
        Calculates Directional Index (DX).

        Implemented without TA-Lib.
        """

        previous_high = self.high.shift(1)
        previous_low = self.low.shift(1)
        previous_close = self.close.shift(1)

        up_move = (
            self.high
            - previous_high
        )

        down_move = (
            previous_low
            - self.low
        )

        plus_dm = pd.Series(
            np.where(
                (up_move > down_move)
                & (up_move > 0),
                up_move,
                0.0,
            ),
            index=self.data.index,
        )

        minus_dm = pd.Series(
            np.where(
                (down_move > up_move)
                & (down_move > 0),
                down_move,
                0.0,
            ),
            index=self.data.index,
        )

        true_range = pd.concat(
            [
                self.high - self.low,
                (
                    self.high
                    - previous_close
                ).abs(),
                (
                    self.low
                    - previous_close
                ).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()

        plus_dm_smoothed = plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()

        minus_dm_smoothed = minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()

        plus_di = (
            100.0
            * plus_dm_smoothed
            / atr
        )

        minus_di = (
            100.0
            * minus_dm_smoothed
            / atr
        )

        denominator = (
            plus_di
            + minus_di
        )

        dx = (
            100.0
            * (
                plus_di
                - minus_di
            ).abs()
            / denominator
        )

        # If both DI values are zero, DX is zero.
        dx = dx.where(
            denominator != 0,
            0.0,
        )

        return dx

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

        pivot = self._calculate_pivots(
            values=self.high,
            period=period,
            mode="high",
        )

        distance = (
            (
                self.close
                - pivot["value"]
            )
            / pivot["value"]
        ) * 100.0

        minutes_away = (
            self.data["timestamp"]
            - pivot["available_timestamp"]
        ).dt.total_seconds() / 60.0

        return pd.DataFrame(
            {
                "pivot_high_distance": distance,
                "pivot_high_minutes_away": minutes_away,
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

        pivot = self._calculate_pivots(
            values=self.low,
            period=period,
            mode="low",
        )

        distance = (
            (
                self.close
                - pivot["value"]
            )
            / pivot["value"]
        ) * 100.0

        minutes_away = (
            self.data["timestamp"]
            - pivot["available_timestamp"]
        ).dt.total_seconds() / 60.0

        return pd.DataFrame(
            {
                "pivot_low_distance": distance,
                "pivot_low_minutes_away": minutes_away,
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
        Finds fully calculated pivots.

        A pivot at bar i becomes available at:

            i + period

        At that exact bar:

            minutes_away = 0

        The newest fully calculated pivot becomes
        the selected pivot.

        No future information is used before the
        pivot becomes fully calculated.
        """

        values_array = values.to_numpy()

        timestamps = self.data["timestamp"]

        pivot_values = np.full(
            len(values_array),
            np.nan,
            dtype=float,
        )

        pivot_available_times = pd.Series(
            pd.NaT,
            index=self.data.index,
            dtype="datetime64[ns, UTC]",
        )

        latest_pivot_value = np.nan

        latest_pivot_available_time = pd.NaT

        for current_index in range(
            len(values_array)
        ):

            candidate_index = (
                current_index - period
            )

            if candidate_index < period:

                pivot_values[
                    current_index
                ] = latest_pivot_value

                pivot_available_times.iloc[
                    current_index
                ] = latest_pivot_available_time

                continue

            left_start = (
                candidate_index - period
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

            left_values = values_array[
                left_start:left_end
            ]

            candidate_value = (
                values_array[
                    candidate_index
                ]
            )

            right_values = values_array[
                right_start:right_end
            ]

            if (
                np.isnan(candidate_value)
                or np.isnan(left_values).any()
                or np.isnan(right_values).any()
            ):

                pivot_values[
                    current_index
                ] = latest_pivot_value

                pivot_available_times.iloc[
                    current_index
                ] = latest_pivot_available_time

                continue

            if mode == "high":

                is_pivot = (
                    candidate_value
                    >= left_values.max()
                    and
                    candidate_value
                    >= right_values.max()
                )

            elif mode == "low":

                is_pivot = (
                    candidate_value
                    <= left_values.min()
                    and
                    candidate_value
                    <= right_values.min()
                )

            else:

                raise ValueError(
                    "mode must be 'high' or 'low'"
                )

            if is_pivot:

                latest_pivot_value = (
                    candidate_value
                )

                # The pivot becomes available NOW,
                # not at the original pivot bar.
                latest_pivot_available_time = (
                    timestamps.iloc[
                        current_index
                    ]
                )

            pivot_values[
                current_index
            ] = latest_pivot_value

            pivot_available_times.iloc[
                current_index
            ] = latest_pivot_available_time

        return pd.DataFrame(
            {
                "value": pivot_values,
                "available_timestamp":
                    pivot_available_times,
            },
            index=self.data.index,
        )
