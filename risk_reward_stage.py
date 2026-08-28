import numpy as np
import pandas as pd
from numba import njit

from cloud_access import (
    save_risk_reward_data,
    log,
)


class RiskRewardStage:

    TRADE_TYPES = ["long", "short"]

    STOP_LOSS_PERCENTAGES = np.array(
        [0.5, 1, 2, 3, 5],
        dtype=np.float64,
    )

    RISK_REWARD_RATIOS = np.array(
        [1, 1.5, 2, 3, 5],
        dtype=np.float64,
    )

    def __init__(self):

        self.trade_types = self.TRADE_TYPES

        self.stop_loss_percentages = (
            self.STOP_LOSS_PERCENTAGES
        )

        self.risk_reward_ratios = (
            self.RISK_REWARD_RATIOS
        )

    # ============================================================
    # MAIN STAGE
    # ============================================================

    def run(
        self,
        symbol,
        close_data,
        high_data,
        low_data,
        timestamps,
    ):
        """
        Risk Reward Stage.

        Process:

        1. Convert timestamps to UTC.
        2. Convert timestamps to New York time.
        3. Keep only data from January 1, 2020 onward.
        4. Convert 1-minute OHLC data to 5-minute OHLC bars.
        5. Every 5-minute bar becomes a possible entry.
        6. Simulate all stop-loss/RR combinations together.
        7. Use NumPy + Numba for the simulation.
        8. Save results separately for each:
               trade_type
               stop_loss_percentage
               risk_reward_ratio

        The Risk Reward Stage does NOT use a training/validation split.
        All available data from January 1, 2020 through the end is used.
        """

        log(
            f"Risk Reward Stage started | "
            f"symbol={symbol}"
        )

        # ========================================================
        # CONVERT INPUTS TO PANDAS SERIES
        # ========================================================

        close_data = pd.Series(
            close_data
        ).reset_index(drop=True)

        high_data = pd.Series(
            high_data
        ).reset_index(drop=True)

        low_data = pd.Series(
            low_data
        ).reset_index(drop=True)

        timestamps = pd.Series(
            timestamps
        ).reset_index(drop=True)

        # ========================================================
        # CONVERT TIMESTAMPS
        #
        # Source timestamps are interpreted as UTC first.
        # They are then converted to New York time.
        # ========================================================

        timestamps = pd.to_datetime(
            timestamps,
            utc=True,
        )

        timestamps = timestamps.dt.tz_convert(
            "America/New_York"
        )

        # ========================================================
        # REMOVE INVALID ROWS
        # ========================================================

        valid_rows = (
            close_data.notna()
            & high_data.notna()
            & low_data.notna()
            & timestamps.notna()
        )

        close_data = close_data.loc[
            valid_rows
        ].reset_index(drop=True)

        high_data = high_data.loc[
            valid_rows
        ].reset_index(drop=True)

        low_data = low_data.loc[
            valid_rows
        ].reset_index(drop=True)

        timestamps = timestamps.loc[
            valid_rows
        ].reset_index(drop=True)

        # ========================================================
        # FILTER TO JANUARY 1, 2020 ONWARD
        #
        # We do NOT require January 1 itself to exist.
        # The first available trading bar on or after Jan 1
        # is used.
        # ========================================================

        start_date = pd.Timestamp(
            "2020-01-01",
            tz="America/New_York",
        )

        date_mask = (
            timestamps >= start_date
        )

        close_data = close_data.loc[
            date_mask
        ].reset_index(drop=True)

        high_data = high_data.loc[
            date_mask
        ].reset_index(drop=True)

        low_data = low_data.loc[
            date_mask
        ].reset_index(drop=True)

        timestamps = timestamps.loc[
            date_mask
        ].reset_index(drop=True)

        if len(close_data) < 2:

            log(
                f"Risk Reward Stage skipped | "
                f"symbol={symbol} | "
                f"reason=not enough data after 2020-01-01"
            )

            return {}

        log(
            f"Risk Reward 2020+ data prepared | "
            f"symbol={symbol} | "
            f"rows={len(close_data):,} | "
            f"start={timestamps.iloc[0]} | "
            f"end={timestamps.iloc[-1]}"
        )

        # ========================================================
        # CONVERT 1-MINUTE DATA TO 5-MINUTE DATA
        # ========================================================

        log(
            f"Risk Reward 5-minute conversion started | "
            f"symbol={symbol}"
        )

        five_minute_data = self._convert_to_5_minutes(
            close_data=close_data,
            high_data=high_data,
            low_data=low_data,
            timestamps=timestamps,
        )

        if len(five_minute_data) < 2:

            log(
                f"Risk Reward Stage skipped | "
                f"symbol={symbol} | "
                f"reason=not enough 5-minute data"
            )

            return {}

        close_5m = five_minute_data[
            "close"
        ].to_numpy(
            dtype=np.float64
        )

        high_5m = five_minute_data[
            "high"
        ].to_numpy(
            dtype=np.float64
        )

        low_5m = five_minute_data[
            "low"
        ].to_numpy(
            dtype=np.float64
        )

        timestamps_5m = five_minute_data[
            "timestamp"
        ]

        log(
            f"Risk Reward 5-minute conversion completed | "
            f"symbol={symbol} | "
            f"rows={len(close_5m):,} | "
            f"start={timestamps_5m.iloc[0]} | "
            f"end={timestamps_5m.iloc[-1]}"
        )

        # ========================================================
        # RESULT CONTAINER
        # ========================================================

        risk_reward_results = {}

        entry_indices = np.arange(
            len(close_5m) - 1,
            dtype=np.int64,
        )

        # ========================================================
        # SIMULATE LONG + SHORT
        # ========================================================

        for trade_type in self.trade_types:

            log(
                f"Risk Reward simulation started | "
                f"symbol={symbol} | "
                f"trade_type={trade_type}"
            )

            is_long = trade_type == "long"

            # ====================================================
            # RUN NUMBA SIMULATION
            #
            # Shape:
            #
            # [stop_loss, ratio, entry]
            #
            # Every combination is calculated in one pass.
            # ====================================================

            exit_indices, exit_prices = _simulate_all_entries(
                close_data=close_5m,
                high_data=high_5m,
                low_data=low_5m,
                stop_loss_percentages=(
                    self.stop_loss_percentages
                ),
                risk_reward_ratios=(
                    self.risk_reward_ratios
                ),
                is_long=is_long,
            )

            # ====================================================
            # CREATE RESULT DICTIONARY
            # ====================================================

            risk_reward_results[
                trade_type
            ] = {}

            # ====================================================
            # CREATE EACH SL / RR DATASET
            # ====================================================

            for sl_index, stop_loss_percentage in enumerate(
                self.stop_loss_percentages
            ):

                risk_reward_results[
                    trade_type
                ][
                    float(stop_loss_percentage)
                ] = {}

                log(
                    f"Risk Reward results started | "
                    f"symbol={symbol} | "
                    f"trade_type={trade_type} | "
                    f"SL={stop_loss_percentage}%"
                )

                for ratio_index, ratio in enumerate(
                    self.risk_reward_ratios
                ):

                    rows = self._create_result_rows(
                        close_data=close_5m,
                        timestamps=timestamps_5m,
                        entry_indices=entry_indices,
                        exit_indices=exit_indices[
                            sl_index,
                            ratio_index,
                        ],
                        exit_prices=exit_prices[
                            sl_index,
                            ratio_index,
                        ],
                        trade_type=trade_type,
                    )

                    risk_reward_data = pd.DataFrame(
                        rows
                    )

                    risk_reward_results[
                        trade_type
                    ][
                        float(stop_loss_percentage)
                    ][
                        float(ratio)
                    ] = risk_reward_data

                    log(
                        f"Risk Reward saving | "
                        f"symbol={symbol} | "
                        f"trade_type={trade_type} | "
                        f"SL={stop_loss_percentage}% | "
                        f"RR={ratio} | "
                        f"rows={len(risk_reward_data):,}"
                    )

                    save_risk_reward_data(
                        symbol=symbol,
                        trade_type=trade_type,
                        stop_loss_percentage=(
                            float(
                                stop_loss_percentage
                            )
                        ),
                        risk_reward_ratio=(
                            float(ratio)
                        ),
                        risk_reward_data=(
                            risk_reward_data
                        ),
                    )

        # ========================================================
        # COMPLETE
        # ========================================================

        log(
            f"Risk Reward Stage completed | "
            f"symbol={symbol}"
        )

        return risk_reward_results

    # ============================================================
    # 1-MINUTE -> 5-MINUTE
    # ============================================================

    @staticmethod
    def _convert_to_5_minutes(
        close_data,
        high_data,
        low_data,
        timestamps,
    ):
        """
        Convert 1-minute OHLC data into standard 5-minute candles.

        Example:

            10:00 -> 10:05 candle
            10:05 -> 10:10 candle
            10:10 -> 10:15 candle

        The timestamp stored for each candle is its CLOSE time.

        Therefore:

            10:00-10:05 candle -> timestamp 10:05
            10:05-10:10 candle -> timestamp 10:10

        The RR trade enters at the candle close and only uses
        subsequent candles for SL/TP detection.
        """

        data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": close_data,
                "high": high_data,
                "low": low_data,
            }
        )

        data = data.set_index(
            "timestamp"
        )

        data = data.sort_index()

        # ------------------------------------------------------------
        # Create standard 5-minute candles.
        #
        # 10:00-10:05 candle contains all available
        # 1-minute bars from 10:00 up to, but not including 10:05.
        #
        # Missing 1-minute bars are allowed.
        # A window is valid as long as at least one
        # valid 1-minute bar exists.
        #
        # The resulting timestamp is shifted to 10:05,
        # representing the candle close / trade entry time.
        # ------------------------------------------------------------

        five_minute = data.resample(
            "5min",
            label="left",
            closed="left",
        ).agg(
            {
                "close": "last",
                "high": "max",
                "low": "min",
            }
        )

        five_minute = five_minute.dropna(
            subset=[
                "close",
                "high",
                "low",
            ]
        )

        # ------------------------------------------------------------
        # Move timestamp from candle OPEN to candle CLOSE.
        # ------------------------------------------------------------

        five_minute.index = (
            five_minute.index
            + pd.Timedelta(minutes=5)
        )

        five_minute = five_minute.reset_index()

        return five_minute

    # ============================================================
    # CREATE RESULT ROWS
    # ============================================================

    @staticmethod
    def _create_result_rows(
        close_data,
        timestamps,
        entry_indices,
        exit_indices,
        exit_prices,
        trade_type,
    ):
        """
        Create the final RR result rows.

        Exit price is the actual price at which the trade
        is considered closed:

            TP hit  -> actual TP price
            SL hit  -> actual SL price
            Dataset ends -> final available candle close
        """

        rows = []

        for position in range(
            len(entry_indices)
        ):

            entry_index = entry_indices[
                position
            ]

            exit_index = exit_indices[
                position
            ]

            entry_price = close_data[
                entry_index
            ]

            exit_price = exit_prices[
                position
            ]

            # ====================================================
            # TRADE RETURN
            # ====================================================

            if trade_type == "long":

                trade_return_percent = (
                    (
                        exit_price
                        / entry_price
                    )
                    - 1.0
                ) * 100.0

            else:

                trade_return_percent = (
                    (
                        entry_price
                        - exit_price
                    )
                    / entry_price
                ) * 100.0

            # ====================================================
            # TIMESTAMPS
            # ====================================================

            start_timestamp = timestamps.iloc[
                entry_index
            ]

            end_timestamp = timestamps.iloc[
                exit_index
            ]

            # ====================================================
            # ELAPSED TIME
            # ====================================================

            time_elapsed_minutes = (
                end_timestamp
                - start_timestamp
            ).total_seconds() / 60.0

            # ====================================================
            # RESULT
            # ====================================================

            rows.append(
                {
                    "start_timestamp": (
                        start_timestamp
                    ),
                    "end_timestamp": (
                        end_timestamp
                    ),
                    "time_elapsed_minutes": (
                        time_elapsed_minutes
                    ),
                    "trade_return_percent": (
                        trade_return_percent
                    ),
                    "entry_price": (
                        entry_price
                    ),
                    "exit_price": (
                        exit_price
                    ),
                }
            )

        return rows

# =================================================================
# NUMBA SIMULATION
# =================================================================

@njit
def _simulate_all_entries(
    close_data,
    high_data,
    low_data,
    stop_loss_percentages,
    risk_reward_ratios,
    is_long,
):
    """
    Simulate every entry using all SL/RR crews together.

    For every entry:

        1. Calculate every actual SL price.
        2. Calculate every actual TP price.
        3. Start with all 25 RR trades active.
        4. Scan forward through the 5-minute candles.
        5. Determine which active barriers are touched.
        6. If an SL is touched, its entire crew is removed.
        7. If a TP is touched, only that RR is removed.
        8. If SL and TP are touched on the same candle,
           the SL wins.
        9. Multiple SLs touched on the same candle all
           remove their respective crews.
        10. If the dataset ends, remaining trades exit
            at the final available close.
    """

    n = len(close_data)

    num_stops = len(
        stop_loss_percentages
    )

    num_ratios = len(
        risk_reward_ratios
    )

    num_entries = n - 1

    # ============================================================
    # OUTPUT ARRAYS
    # ============================================================

    exit_indices = np.empty(
        (
            num_stops,
            num_ratios,
            num_entries,
        ),
        dtype=np.int64,
    )

    exit_prices = np.empty(
        (
            num_stops,
            num_ratios,
            num_entries,
        ),
        dtype=np.float64,
    )

    # ============================================================
    # EVERY ENTRY
    # ============================================================

    for entry_index in range(
        num_entries
    ):

        entry_price = close_data[
            entry_index
        ]

        # ========================================================
        # ACTIVE CREWS
        #
        # active[stop, ratio]
        #
        # True = this RR is still active.
        # ========================================================

        active = np.ones(
            (
                num_stops,
                num_ratios,
            ),
            dtype=np.bool_,
        )

        # ========================================================
        # CALCULATE ALL ACTUAL BARRIER PRICES
        # ========================================================

        stop_prices = np.empty(
            num_stops,
            dtype=np.float64,
        )

        take_profit_prices = np.empty(
            (
                num_stops,
                num_ratios,
            ),
            dtype=np.float64,
        )

        for stop_index in range(
            num_stops
        ):

            sl = (
                stop_loss_percentages[
                    stop_index
                ]
                / 100.0
            )

            # ----------------------------------------------------
            # STOP LOSS
            # ----------------------------------------------------

            if is_long:

                stop_prices[
                    stop_index
                ] = (
                    entry_price
                    * (1.0 - sl)
                )

            else:

                stop_prices[
                    stop_index
                ] = (
                    entry_price
                    * (1.0 + sl)
                )

            # ----------------------------------------------------
            # TAKE PROFITS
            # ----------------------------------------------------

            for ratio_index in range(
                num_ratios
            ):

                ratio = (
                    risk_reward_ratios[
                        ratio_index
                    ]
                )

                tp_percent = (
                    sl * ratio
                )

                if is_long:

                    take_profit_prices[
                        stop_index,
                        ratio_index,
                    ] = (
                        entry_price
                        * (
                            1.0
                            + tp_percent
                        )
                    )

                else:

                    take_profit_prices[
                        stop_index,
                        ratio_index,
                    ] = (
                        entry_price
                        * (
                            1.0
                            - tp_percent
                        )
                    )

        # ========================================================
        # NUMBER OF ACTIVE RR TRADES
        # ========================================================

        remaining = (
            num_stops
            * num_ratios
        )

        # ========================================================
        # SCAN FORWARD THROUGH 5-MINUTE CANDLES
        # ========================================================

        for bar_index in range(
            entry_index + 1,
            n,
        ):

            if remaining == 0:
                break

            current_high = high_data[
                bar_index
            ]

            current_low = low_data[
                bar_index
            ]

            # ====================================================
            # FIRST PASS:
            #
            # Determine which SL crews were hit.
            #
            # We do this FIRST because:
            #
            # SL + TP on same candle = SL wins.
            # ====================================================

            stop_hit = np.zeros(
                num_stops,
                dtype=np.bool_,
            )

            for stop_index in range(
                num_stops
            ):

                # ------------------------------------------------
                # Skip completely inactive crews.
                # ------------------------------------------------

                crew_active = False

                for ratio_index in range(
                    num_ratios
                ):

                    if active[
                        stop_index,
                        ratio_index,
                    ]:

                        crew_active = True
                        break

                if not crew_active:
                    continue

                # ------------------------------------------------
                # Check SL.
                # ------------------------------------------------

                if is_long:

                    if (
                        current_low
                        <= stop_prices[
                            stop_index
                        ]
                    ):

                        stop_hit[
                            stop_index
                        ] = True

                else:

                    if (
                        current_high
                        >= stop_prices[
                            stop_index
                        ]
                    ):

                        stop_hit[
                            stop_index
                        ] = True

            # ====================================================
            # REMOVE ALL SL CREWS HIT ON THIS CANDLE
            #
            # This handles:
            #
            # SL 0.5% hit
            # SL 1.0% hit
            #
            # on the same candle.
            #
            # Both crews are removed.
            # ====================================================

            for stop_index in range(
                num_stops
            ):

                if not stop_hit[
                    stop_index
                ]:

                    continue

                for ratio_index in range(
                    num_ratios
                ):

                    if active[
                        stop_index,
                        ratio_index,
                    ]:

                        active[
                            stop_index,
                            ratio_index,
                        ] = False

                        exit_indices[
                            stop_index,
                            ratio_index,
                            entry_index,
                        ] = bar_index

                        exit_prices[
                            stop_index,
                            ratio_index,
                            entry_index,
                        ] = stop_prices[
                            stop_index
                        ]

                        remaining -= 1

            # ====================================================
            # SECOND PASS:
            #
            # Check TPs.
            #
            # Any TP belonging to a crew whose SL was hit
            # is already inactive, so it cannot win.
            #
            # This automatically gives SL priority when both
            # SL and TP occur in the same candle.
            # ====================================================

            for stop_index in range(
                num_stops
            ):

                # ------------------------------------------------
                # If the SL crew was hit, its entire crew
                # has already been removed.
                # ------------------------------------------------

                if stop_hit[
                    stop_index
                ]:

                    continue

                # ------------------------------------------------
                # Check every active TP.
                #
                # Multiple TP barriers can be crossed by the
                # same candle, so every active RR is checked.
                # ------------------------------------------------

                for ratio_index in range(
                    num_ratios
                ):

                    if not active[
                        stop_index,
                        ratio_index,
                    ]:

                        continue

                    tp_reached = False

                    if is_long:

                        if (
                            current_high
                            >= take_profit_prices[
                                stop_index,
                                ratio_index,
                            ]
                        ):

                            tp_reached = True

                    else:

                        if (
                            current_low
                            <= take_profit_prices[
                                stop_index,
                                ratio_index,
                            ]
                        ):

                            tp_reached = True

                    if tp_reached:

                        active[
                            stop_index,
                            ratio_index,
                        ] = False

                        exit_indices[
                            stop_index,
                            ratio_index,
                            entry_index,
                        ] = bar_index

                        exit_prices[
                            stop_index,
                            ratio_index,
                            entry_index,
                        ] = take_profit_prices[
                            stop_index,
                            ratio_index,
                        ]

                        remaining -= 1

            # ====================================================
            # CONTINUE TO NEXT BAR
            # ====================================================

        # ========================================================
        # DATASET ENDED
        #
        # Any RR that never reached SL or TP exits at the
        # final available 5-minute close.
        # ========================================================

        final_index = n - 1

        for stop_index in range(
            num_stops
        ):

            for ratio_index in range(
                num_ratios
            ):

                if active[
                    stop_index,
                    ratio_index,
                ]:

                    exit_indices[
                        stop_index,
                        ratio_index,
                        entry_index,
                    ] = final_index

                    exit_prices[
                        stop_index,
                        ratio_index,
                        entry_index,
                    ] = close_data[
                        final_index
                    ]
    return (
        exit_indices,
        exit_prices,
    )