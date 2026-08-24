import random
import pandas as pd

from cloud_access import (
    save_risk_reward_data,
    log,
)


class RiskRewardStage:

    TRADE_TYPES = ["long", "short"]

    STOP_LOSS_PERCENTAGES = [
        0.5,
        1,
        2,
        3,
        5,
    ]

    RISK_REWARD_RATIOS = [
        1,
        1.5,
        2,
        3,
        5,
    ]

    MAX_DELAY_BARS = 10

    def __init__(self):
        self.trade_types = self.TRADE_TYPES
        self.stop_loss_percentages = self.STOP_LOSS_PERCENTAGES
        self.risk_reward_ratios = sorted(
            self.RISK_REWARD_RATIOS
        )

        self.entry_price = None
        self.trade_results = None
        self.delay_bars = None
        self.risk_reward_data = None

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
        Run all long/short, stop-loss, and risk/reward simulations.

        close_data:
            Close price for every bar.

        high_data:
            High price for every bar.

        low_data:
            Low price for every bar.

        timestamps:
            Timestamp for every bar.

        Rows containing missing values in any required column
        are skipped completely.
        """

        log(
            f"Risk Reward Stage started | "
            f"symbol={symbol}"
        )

        risk_reward_results = {}

        # ============================================================
        # CONVERT INPUTS TO SERIES
        # ============================================================

        close_data = pd.Series(
            close_data
        ).reset_index(drop=True)

        high_data = pd.Series(
            high_data
        ).reset_index(drop=True)

        low_data = pd.Series(
            low_data
        ).reset_index(drop=True)

        timestamps = pd.to_datetime(
            pd.Series(timestamps),
            utc=True,
        ).reset_index(drop=True)

        # ============================================================
        # REMOVE INVALID ROWS
        # ============================================================

        valid_rows = (
            close_data.notna()
            & high_data.notna()
            & low_data.notna()
            & timestamps.notna()
        )

        close_data = close_data[valid_rows].reset_index(
            drop=True
        )

        high_data = high_data[valid_rows].reset_index(
            drop=True
        )

        low_data = low_data[valid_rows].reset_index(
            drop=True
        )

        timestamps = timestamps[valid_rows].reset_index(
            drop=True
        )

        # ============================================================
        # SIMULATE EACH TRADE TYPE
        # ============================================================

        for trade_type in self.trade_types:

            log(
                f"Risk Reward processing | "
                f"symbol={symbol} | "
                f"trade_type={trade_type}"
            )

            risk_reward_results[trade_type] = {}
            
            for stop_loss_percentage in (
                self.stop_loss_percentages
            ):
                risk_reward_results[
                    trade_type
                ][
                    stop_loss_percentage
                ] = {}

                results_by_ratio = {
                    ratio: []
                    for ratio in self.risk_reward_ratios
                }

                # ====================================================
                # EVERY BAR IS AN ENTRY
                # ====================================================

                for entry_index in range(
                    len(close_data) - 1
                ):

                    entry_price = close_data.iloc[
                        entry_index
                    ]

                    self.entry_price = entry_price

                    # =================================================
                    # FIND ORIGINAL EXIT FOR EACH RR
                    # =================================================

                    outcomes = self._simulate_entry(
                        close_data=close_data,
                        high_data=high_data,
                        low_data=low_data,
                        timestamps=timestamps,
                        entry_index=entry_index,
                        entry_price=entry_price,
                        trade_type=trade_type,
                        stop_loss_percentage=(
                            stop_loss_percentage
                        ),
                    )

                    # =================================================
                    # CREATE RESULT FOR EACH RR
                    # =================================================

                    for ratio in self.risk_reward_ratios:

                        outcome = outcomes[ratio]

                        result = self._create_result_row(
                            close_data=close_data,
                            timestamps=timestamps,
                            entry_index=entry_index,
                            entry_price=entry_price,
                            trade_type=trade_type,
                            outcome=outcome,
                        )

                        results_by_ratio[
                            ratio
                        ].append(result)

                # ====================================================
                # SAVE EACH RR
                # ====================================================

                for ratio, rows in (
                    results_by_ratio.items()
                ):

                    risk_reward_data = pd.DataFrame(
                        rows
                    )

                    risk_reward_results[
                        trade_type
                    ][
                        stop_loss_percentage
                    ][
                        ratio
                    ] = risk_reward_data

                    save_risk_reward_data(
                        symbol=symbol,
                        trade_type=trade_type,
                        stop_loss_percentage=(
                            stop_loss_percentage
                        ),
                        risk_reward_ratio=ratio,
                        risk_reward_data=(
                            risk_reward_data
                        ),
                    )
        log(
            f"Risk Reward Stage completed | "
            f"symbol={symbol}"
        )

        return risk_reward_results

        
    # ============================================================
    # SIMULATE ONE ENTRY
    # ============================================================

    def _simulate_entry(
        self,
        close_data,
        high_data,
        low_data,
        timestamps,
        entry_index,
        entry_price,
        trade_type,
        stop_loss_percentage,
    ):
        """
        Simulate one trade entry.

        The trade starts at the entry bar's closing price.

        For every following bar:

            1. Check stop loss using the bar's low/high.
            2. Check take profit using the bar's high/low.
            3. If both are reached in the same bar,
            stop loss is assumed to happen first.
            4. If take profit is reached, that RR is completed
            and simulation continues toward the next RR.
            5. If stop loss is reached, every remaining RR gets
            the same stop-loss exit.
            6. If the dataset ends, all remaining RR values are
            recorded as losses.
        """

        ratios = self.risk_reward_ratios

        outcomes = {}

        # ============================================================
        # STOP LOSS AS DECIMAL
        # ============================================================

        stop_loss_percentage = (
            stop_loss_percentage / 100.0
        )

        # ============================================================
        # CALCULATE STOP LOSS PRICE
        # ============================================================

        if trade_type == "long":

            stop_loss = (
                entry_price
                * (1.0 - stop_loss_percentage)
            )

        else:

            stop_loss = (
                entry_price
                * (1.0 + stop_loss_percentage)
            )

        # ============================================================
        # START WITH LOWEST RR
        # ============================================================

        current_ratio_index = 0

        # ============================================================
        # CHECK EVERY FOLLOWING BAR
        # ============================================================

        for bar_index in range(
            entry_index + 1,
            len(close_data),
        ):

            current_high = high_data.iloc[
                bar_index
            ]

            current_low = low_data.iloc[
                bar_index
            ]

            # ========================================================
            # SAFETY: SKIP INVALID BAR
            # ========================================================

            if (
                pd.isna(current_high)
                or pd.isna(current_low)
            ):
                continue

            # ========================================================
            # CURRENT RATIO
            # ========================================================

            current_ratio = ratios[
                current_ratio_index
            ]

            # ========================================================
            # TAKE PROFIT PRICE
            # ========================================================

            take_profit_percentage = (
                stop_loss_percentage
                * current_ratio
            )

            if trade_type == "long":

                take_profit = (
                    entry_price
                    * (
                        1.0
                        + take_profit_percentage
                    )
                )

            else:

                take_profit = (
                    entry_price
                    * (
                        1.0
                        - take_profit_percentage
                    )
                )

            # ========================================================
            # CHECK STOP LOSS
            # ========================================================

            if trade_type == "long":

                stop_reached = (
                    current_low <= stop_loss
                )

            else:

                stop_reached = (
                    current_high >= stop_loss
                )

            # ========================================================
            # CHECK TAKE PROFIT
            # ========================================================

            if trade_type == "long":

                take_profit_reached = (
                    current_high >= take_profit
                )

            else:

                take_profit_reached = (
                    current_low <= take_profit
                )

            # ========================================================
            # BOTH HIT IN SAME BAR
            # ========================================================

            if (
                stop_reached
                and take_profit_reached
            ):

                # Stop loss always wins when both
                # are reached in the same bar.

                for remaining_index in range(
                    current_ratio_index,
                    len(ratios),
                ):

                    ratio = ratios[
                        remaining_index
                    ]

                    outcomes[ratio] = {
                        "exit_index": bar_index,
                        "outcome": "stop_loss",
                        "ratio_completed": False,
                    }

                break

            # ========================================================
            # STOP LOSS ONLY
            # ========================================================

            if stop_reached:

                for remaining_index in range(
                    current_ratio_index,
                    len(ratios),
                ):

                    ratio = ratios[
                        remaining_index
                    ]

                    outcomes[ratio] = {
                        "exit_index": bar_index,
                        "outcome": "stop_loss",
                        "ratio_completed": False,
                    }

                break

            # ========================================================
            # TAKE PROFIT ONLY
            # ========================================================

            if take_profit_reached:

                outcomes[current_ratio] = {
                    "exit_index": bar_index,
                    "outcome": "take_profit",
                    "ratio_completed": True,
                }

                current_ratio_index += 1

                # ====================================================
                # ALL RATIOS COMPLETED
                # ====================================================

                if (
                    current_ratio_index
                    >= len(ratios)
                ):
                    break

        # ============================================================
        # DATASET ENDED
        # ============================================================

        else:

            final_index = len(close_data) - 1

            for remaining_index in range(
                current_ratio_index,
                len(ratios),
            ):

                ratio = ratios[
                    remaining_index
                ]

                outcomes[ratio] = {
                    "exit_index": final_index,
                    "outcome": "loss",
                    "ratio_completed": False,
                }

        # ============================================================
        # SAFETY FALLBACK
        # ============================================================

        final_index = len(close_data) - 1

        for ratio in ratios:

            if ratio not in outcomes:

                outcomes[ratio] = {
                    "exit_index": final_index,
                    "outcome": "loss",
                    "ratio_completed": False,
                }

        return outcomes

    # ============================================================
    # CREATE RESULT ROW
    # ============================================================

    def _create_result_row(
        self,
        close_data,
        timestamps,
        entry_index,
        entry_price,
        trade_type,
        outcome,
    ):
        """
        Apply the independently randomized exit delay and
        calculate the stored simulation statistics.
        """

        original_exit_index = outcome[
            "exit_index"
        ]

        # --------------------------------------------------------
        # Random delay
        # --------------------------------------------------------

        delay_bars = random.randint(
            0,
            self.MAX_DELAY_BARS,
        )

        delayed_exit_index = min(
            original_exit_index + delay_bars,
            len(close_data) - 1,
        )

        self.delay_bars = delay_bars

        # --------------------------------------------------------
        # Return series
        # --------------------------------------------------------

        returns = []

        equity = 1.0

        max_equity = equity
        max_drawdown = 0.0

        for index in range(
            entry_index + 1,
            delayed_exit_index + 1,
        ):

            previous_close = close_data.iloc[
                index - 1
            ]

            current_close = close_data.iloc[
                index
            ]

            price_return = (
                current_close / previous_close
            ) - 1.0

            # Short profits when price decreases.
            if trade_type == "short":
                price_return = -price_return

            returns.append(
                price_return
            )

            equity *= (
                1.0 + price_return
            )

            max_equity = max(
                max_equity,
                equity,
            )

            if max_equity != 0:

                drawdown = (
                    (max_equity - equity)
                    / max_equity
                )

                max_drawdown = max(
                    max_drawdown,
                    drawdown,
                )

        # --------------------------------------------------------
        # Statistics
        # --------------------------------------------------------

        returns_series = pd.Series(
            returns,
            dtype=float,
        )

        n = len(
            returns_series
        )

        sum_r = returns_series.sum()

        sum_r2 = (
            returns_series ** 2
        ).sum()

        negative_returns = returns_series[
            returns_series < 0
        ]

        sum_d2 = (
            negative_returns ** 2
        ).sum()

        sum_r3 = (
            returns_series ** 3
        ).sum()

        # --------------------------------------------------------
        # Timestamps
        # --------------------------------------------------------

        start_timestamp = timestamps.iloc[
            entry_index
        ]

        end_timestamp = timestamps.iloc[
            delayed_exit_index
        ]

        # --------------------------------------------------------
        # Market session
        # --------------------------------------------------------

        entry_market_type = (
            self._get_market_type(
                start_timestamp
            )
        )

        held_overnight = (
            start_timestamp.tz_convert(
                "America/New_York"
            ).date()
            !=
            end_timestamp.tz_convert(
                "America/New_York"
            ).date()
        )

        return {
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "N": n,
            "Sum_R": sum_r,
            "Sum_R2": sum_r2,
            "Sum_D2": sum_d2,
            "Sum_R3": sum_r3,
            "Max_Equity": max_equity,
            "End_Equity": equity,
            "Max_DD": max_drawdown,
            "entry_market_type": entry_market_type,
            "held_overnight": held_overnight,
            "entry_close": entry_price,
        }

    # ============================================================
    # MARKET SESSION
    # ============================================================

    @staticmethod
    def _get_market_type(timestamp):
        """
        Determine market session using New York local time.

        -1 = premarket
         0 = market hours
         1 = after hours
        """

        local_time = timestamp.tz_convert(
            "America/New_York"
        )

        time = local_time.time()

        premarket_start = pd.Timestamp(
            "04:00"
        ).time()

        market_start = pd.Timestamp(
            "09:30"
        ).time()

        market_end = pd.Timestamp(
            "16:00"
        ).time()

        after_hours_end = pd.Timestamp(
            "20:00"
        ).time()

        if (
            premarket_start
            <= time
            < market_start
        ):
            return -1

        if (
            market_start
            <= time
            < market_end
        ):
            return 0

        if (
            market_end
            <= time
            < after_hours_end
        ):
            return 1

        return -1