from input_stage import InputStage
from risk_reward_stage import RiskRewardStage
from state_space_stage import StateSpaceStage

from cloud_access import (
    log,
    log_error,
)


# ============================================================
# STOCK CONFIGURATION
# ============================================================

STOCK_SYMBOL = "DELL"


# ============================================================
# BIN COUNT RULES
# ============================================================

def get_bin_count(input_name):
    """
    Return the configured bin count for an input.

    These are the configured bin counts passed into
    StatisticsAPI.calculate_bin_configuration().

    The final bin count may become larger because the
    IQR-based calculation expands the bins to cover
    the complete min-to-max range.
    """

    # --------------------------------------------------------
    # Dollar volume
    # --------------------------------------------------------

    if "dollar_volume" in input_name:
        return 5

    # --------------------------------------------------------
    # Minutes
    # --------------------------------------------------------

    if (
        input_name == "current_time"
        or "minutes_away" in input_name
    ):
        return 4

    # --------------------------------------------------------
    # Percentage values
    # --------------------------------------------------------

    if (
        "ema_distance" in input_name
        or "dema_distance" in input_name
        or "vwap_distance" in input_name
        or "roc" in input_name
        or "return_standard_deviation" in input_name
        or "normalized_atr" in input_name
        or "pivot_high_distance" in input_name
        or "pivot_low_distance" in input_name
    ):
        return 3

    # --------------------------------------------------------
    # Everything else
    # --------------------------------------------------------

    return 2


def create_bin_counts(input_statistics):
    """
    Create the bin-count configuration automatically
    from the input statistics table.
    """

    bin_counts = {}

    for input_name in input_statistics["input"]:

        bin_counts[input_name] = (
            get_bin_count(input_name)
        )

    return bin_counts


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(symbol):
    """
    Run the complete pipeline for one stock.

    Stages:
        1. InputStage
        2. RiskRewardStage
        3. StateSpaceStage
    """

    log(
        f"START STOCK | SYMBOL={symbol}"
    )

    # ========================================================
    # 1. INPUT STAGE
    # ========================================================

    log(
        f"STAGE START | SYMBOL={symbol} | STAGE=InputStage"
    )

    try:

        input_stage = InputStage()

        input_result = input_stage.run(
            symbol
        )

    except Exception as error:

        log_error(
            stage="InputStage",
            symbol=symbol,
            error=error,
        )

        raise

    log(
        f"STAGE COMPLETE | "
        f"SYMBOL={symbol} | "
        f"STAGE=InputStage"
    )

    # ========================================================
    # GET INPUT RESULTS
    # ========================================================

    input_data = input_result[
        "input_data"
    ]

    input_statistics = input_result[
        "statistics"
    ]

    close_data = input_result[
        "close_data"
    ]

    high_data = input_result[
        "high_data"
    ]

    low_data = input_result[
        "low_data"
    ]

    timestamps = input_result[
        "timestamps"
    ]

    # ========================================================
    # CREATE BIN COUNTS
    # ========================================================

    bin_counts = create_bin_counts(
        input_statistics
    )

    log(
        f"BIN CONFIGURATION CREATED | "
        f"SYMBOL={symbol} | "
        f"INPUTS={len(bin_counts)}"
    )

    # ========================================================
    # 2. RISK / REWARD STAGE
    # ========================================================

    log(
        f"STAGE START | "
        f"SYMBOL={symbol} | "
        f"STAGE=RiskRewardStage"
    )

    try:

        risk_reward_stage = RiskRewardStage()

        risk_reward_results = (
            risk_reward_stage.run(
                symbol=symbol,
                close_data=close_data,
                high_data=high_data,
                low_data=low_data,
                timestamps=timestamps,
            )
        )

    except Exception as error:

        log_error(
            stage="RiskRewardStage",
            symbol=symbol,
            error=error,
        )

        raise

    log(
        f"STAGE COMPLETE | "
        f"SYMBOL={symbol} | "
        f"STAGE=RiskRewardStage"
    )

    # ========================================================
    # 3. STATE SPACE STAGE
    # ========================================================

    log(
        f"STAGE START | "
        f"SYMBOL={symbol} | "
        f"STAGE=StateSpaceStage"
    )

    try:

        state_space_stage = StateSpaceStage()

        total_combinations = 0

        # ----------------------------------------------------
        # Trade types
        # ----------------------------------------------------

        for trade_type in (
            risk_reward_stage.trade_types
        ):

            # ------------------------------------------------
            # Stop losses
            # ------------------------------------------------

            for stop_loss_percentage in (
                risk_reward_stage.stop_loss_percentages
            ):

                # --------------------------------------------
                # Risk/reward ratios
                # --------------------------------------------

                for risk_reward_ratio in (
                    risk_reward_stage.risk_reward_ratios
                ):

                    total_combinations += 1

                    log(
                        f"GRID START | "
                        f"SYMBOL={symbol} | "
                        f"TRADE_TYPE={trade_type} | "
                        f"SL={stop_loss_percentage} | "
                        f"RR={risk_reward_ratio}"
                    )

                    # ----------------------------------------
                    # Get corresponding RR data
                    # ----------------------------------------

                    risk_reward_data = (
                        risk_reward_results[
                            trade_type
                        ][
                            stop_loss_percentage
                        ][
                            risk_reward_ratio
                        ]
                    )

                    # ----------------------------------------
                    # Run state space
                    # ----------------------------------------

                    state_space_stage.run(
                        symbol=symbol,
                        trade_type=trade_type,
                        stop_loss_percentage=(
                            stop_loss_percentage
                        ),
                        risk_reward_ratio=(
                            risk_reward_ratio
                        ),
                        input_data=input_data,
                        input_statistics=(
                            input_statistics
                        ),
                        risk_reward_data=(
                            risk_reward_data
                        ),
                        bin_counts=bin_counts,
                    )

                    log(
                        f"GRID COMPLETE | "
                        f"SYMBOL={symbol} | "
                        f"TRADE_TYPE={trade_type} | "
                        f"SL={stop_loss_percentage} | "
                        f"RR={risk_reward_ratio}"
                    )

    except Exception as error:

        log_error(
            stage="StateSpaceStage",
            symbol=symbol,
            error=error,
            context=(
                f"trade_type={trade_type}, "
                f"stop_loss_percentage="
                f"{stop_loss_percentage}, "
                f"risk_reward_ratio="
                f"{risk_reward_ratio}"
            ),
        )

        raise

    log(
        f"STAGE COMPLETE | "
        f"SYMBOL={symbol} | "
        f"STAGE=StateSpaceStage | "
        f"COMBINATIONS={total_combinations}"
    )

    # ========================================================
    # STOCK COMPLETE
    # ========================================================

    log(
        f"STOCK COMPLETE | SYMBOL={symbol}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log(
        f"PROGRAM STARTED | SYMBOL={STOCK_SYMBOL}"
    )

    try:

        process_stock(
            STOCK_SYMBOL
        )

    except Exception:

        log(
            f"STOCK FAILED | SYMBOL={STOCK_SYMBOL}"
        )

        raise

    log(
        f"PROGRAM STOPPED | SYMBOL={STOCK_SYMBOL}"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
