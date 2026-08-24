import threading

from flask import Flask, jsonify

from pyngrok import ngrok

from cloud_access import (
    configure_logging,
    get_stock_symbols,
    get_completed_symbols,
    mark_stock_completed,
    log,
)


# ============================================================
# SERVER
# ============================================================

app = Flask(__name__)


# ============================================================
# VARIABLES
# ============================================================

stock_symbols = []

completed_symbols = set()

next_stock_index = 0

stock_request_count = 0

stock_request_lock = threading.Lock()


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_server():

    global stock_symbols
    global completed_symbols
    global next_stock_index
    global stock_request_count

    configure_logging(
        "SERVER"
    )

    stock_symbols = (
        get_stock_symbols()
    )

    completed_symbols = set(
        get_completed_symbols()
    )

    next_stock_index = 0

    stock_request_count = 0

    log(
        f"SERVER initialized | "
        f"total_stocks={len(stock_symbols)} | "
        f"completed={len(completed_symbols)} | "
        f"remaining={len(stock_symbols) - len(completed_symbols)}"
    )


# ============================================================
# REQUEST NEXT STOCK
# ============================================================

@app.get("/next-stock")
def request_next_stock():

    global next_stock_index
    global stock_request_count

    with stock_request_lock:

        stock_request_count += 1

        while (
            next_stock_index
            < len(stock_symbols)
        ):

            symbol = (
                stock_symbols[
                    next_stock_index
                ]
            )

            next_stock_index += 1

            if symbol in completed_symbols:
                continue

            completed_count = (
                len(completed_symbols)
            )

            remaining_count = (
                len(stock_symbols)
                - completed_count
            )

            log(
                f"Stock requested | "
                f"symbol={symbol} | "
                f"request_count={stock_request_count} | "
                f"completed={completed_count}/{len(stock_symbols)} | "
                f"remaining={remaining_count}"
            )

            return jsonify({
                "symbol": symbol
            })

        log(
            f"No stock available | "
            f"request_count={stock_request_count} | "
            f"completed={len(completed_symbols)}/{len(stock_symbols)} | "
            f"remaining={len(stock_symbols) - len(completed_symbols)}"
        )

        return jsonify({
            "symbol": None
        })


# ============================================================
# REPORT STOCK COMPLETED
# ============================================================

@app.post("/stock-completed/<symbol>")
def report_stock_completed(symbol):

    with stock_request_lock:

        if symbol not in completed_symbols:

            mark_stock_completed(
                symbol
            )

            completed_symbols.add(
                symbol
            )

        completed_count = (
            len(completed_symbols)
        )

        remaining_count = (
            len(stock_symbols)
            - completed_count
        )

        log(
            f"Stock completed | "
            f"symbol={symbol} | "
            f"completed={completed_count}/{len(stock_symbols)} | "
            f"remaining={remaining_count}"
        )

    return jsonify({
        "symbol": symbol,
        "completed": True,
    })


# ============================================================
# SERVER STATUS
# ============================================================

@app.get("/status")
def server_status():

    with stock_request_lock:

        completed_count = (
            len(completed_symbols)
        )

        remaining_count = (
            len(stock_symbols)
            - completed_count
        )

        return jsonify({
            "server": "SERVER",
            "total_stocks": len(stock_symbols),
            "request_count": stock_request_count,
            "completed": completed_count,
            "remaining": remaining_count,
        })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    initialize_server()

    # --------------------------------------------------------
    # START NGROK
    # --------------------------------------------------------

    public_url = ngrok.connect(
        5000,
        bind_tls=True,
    )

    public_url = str(
        public_url
    )

    print()
    print("=" * 60)
    print("SERVER STARTED")
    print("=" * 60)
    print()

    print(
        f"NGROK SERVER URL:"
    )

    print()

    print(
        public_url
    )

    print()

    print(
        "Put this URL into SERVER_URL "
        "on the pod computers."
    )

    print()

    print("=" * 60)
    print()

    log(
        f"NGROK server started | "
        f"url={public_url}"
    )

    # --------------------------------------------------------
    # START FLASK
    # --------------------------------------------------------

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
    )