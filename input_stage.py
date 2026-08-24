import pandas as pd

from indicator_api import IndicatorAPI
from statistics_api import StatisticsAPI

from cloud_access import (
    download_stock_data,
    save_input_data,
    save_input_statistics,
    log,
)

class InputStage:

    def __init__(self):
        self.raw_stock_data = None
        self.input_data = None
        self.statistics = None
        self.close_data = None

    def run(self, symbol):
        log(
            f"Input Stage started | "
            f"symbol={symbol}"
        )
        # ========================================================
        # 1. LOAD
        # ========================================================

        self.raw_stock_data = download_stock_data(
            symbol
        )

        # IndicatorAPI sorts the stock data chronologically.
        indicator_api = IndicatorAPI(
            self.raw_stock_data
        )

        # Use the exact same sorted data used by IndicatorAPI.
        processed_data = indicator_api.data

        # ========================================================
        # 2. CALCULATE INPUTS
        # ========================================================

        input_data = pd.DataFrame(
            index=processed_data.index
        )

        # --------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------

        input_data["timestamp"] = (
            processed_data["timestamp"]
        )

        # --------------------------------------------------------
        # Current Time
        # --------------------------------------------------------

        input_data["current_time"] = (
            indicator_api.calculate_current_time()
        )

        # --------------------------------------------------------
        # Period-Based Inputs
        # --------------------------------------------------------

        periods = IndicatorAPI.PERIODS

        for period in periods:

            # ----------------------------------------------------
            # Dollar Volume
            # ----------------------------------------------------

            input_data[
                f"dollar_volume_{period}"
            ] = indicator_api.calculate_dollar_volume(
                period
            )

            # ----------------------------------------------------
            # EMA Distance
            # ----------------------------------------------------

            input_data[
                f"ema_distance_{period}"
            ] = indicator_api.calculate_ema_distance(
                period
            )

            # ----------------------------------------------------
            # DEMA Distance
            # ----------------------------------------------------

            input_data[
                f"dema_distance_{period}"
            ] = indicator_api.calculate_dema_distance(
                period
            )

            # ----------------------------------------------------
            # VWAP Distance
            # ----------------------------------------------------

            input_data[
                f"vwap_distance_{period}"
            ] = indicator_api.calculate_vwap_distance()

            # ----------------------------------------------------
            # RSI
            # ----------------------------------------------------

            input_data[
                f"rsi_{period}"
            ] = indicator_api.calculate_rsi(
                period
            )

            # ----------------------------------------------------
            # ROC
            # ----------------------------------------------------

            input_data[
                f"roc_{period}"
            ] = indicator_api.calculate_roc(
                period
            )

            # ----------------------------------------------------
            # Return Standard Deviation
            # ----------------------------------------------------

            input_data[
                f"return_standard_deviation_{period}"
            ] = (
                indicator_api
                .calculate_return_standard_deviation(
                    period
                )
            )

            # ----------------------------------------------------
            # Normalized ATR
            # ----------------------------------------------------

            input_data[
                f"normalized_atr_{period}"
            ] = (
                indicator_api
                .calculate_normalized_atr(
                    period
                )
            )

            # ----------------------------------------------------
            # DX
            # ----------------------------------------------------

            input_data[
                f"dx_{period}"
            ] = indicator_api.calculate_dx(
                period
            )

            # ----------------------------------------------------
            # Pivot High
            # ----------------------------------------------------

            pivot_high = (
                indicator_api.calculate_pivot_high(
                    period
                )
            )

            input_data[
                f"pivot_high_distance_{period}"
            ] = pivot_high[
                "pivot_high_distance"
            ]

            input_data[
                f"pivot_high_minutes_away_{period}"
            ] = pivot_high[
                "pivot_high_minutes_away"
            ]

            # ----------------------------------------------------
            # Pivot Low
            # ----------------------------------------------------

            pivot_low = (
                indicator_api.calculate_pivot_low(
                    period
                )
            )

            input_data[
                f"pivot_low_distance_{period}"
            ] = pivot_low[
                "pivot_low_distance"
            ]

            input_data[
                f"pivot_low_minutes_away_{period}"
            ] = pivot_low[
                "pivot_low_minutes_away"
            ]

        self.input_data = input_data

        # ========================================================
        # 3. STORE CLOSE DATA
        # ========================================================

        self.close_data = (
            processed_data["close"].copy()
        )

        # ========================================================
        # 4. CALCULATE INPUT STATISTICS
        # ========================================================

        statistics_rows = []

        for input_name in self.input_data.columns:

            # Timestamp is not a numerical input.
            if input_name == "timestamp":
                continue

            input_values = self.input_data[
                input_name
            ]

            result = (
                StatisticsAPI.calculate_statistics(
                    input_values
                )
            )

            statistics_rows.append({
                "input": input_name,

                "min": result["min"],
                "q1": result["q1"],
                "median": result["median"],
                "q3": result["q3"],
                "max": result["max"],

                "sum": result["sum"],
                "count": result["count"],
            })

        self.statistics = pd.DataFrame(
            statistics_rows
        )

        # ========================================================
        # 5. SAVE
        # ========================================================

        save_input_data(
            symbol,
            self.input_data,
        )

        save_input_statistics(
            symbol,
            self.statistics,
        )
    
        # ========================================================
        # 6. RETURN
        # ========================================================

        log(
            f"Input Stage completed | "
            f"symbol={symbol} | "
            f"rows={len(self.input_data)} | "
            f"inputs={len(self.input_data.columns) - 1}"
        )

        return {
            "input_data": self.input_data,
            "statistics": self.statistics,
            "close_data": self.close_data,
            "high_data": processed_data["high"].copy(),
            "low_data": processed_data["low"].copy(),
            "timestamps": processed_data["timestamp"].copy(),
        }