#!/bin/bash

set -e

echo "========================================"
echo "Updating Ubuntu..."
echo "========================================"

apt update
apt install -y python3 python3-pip wget

echo "========================================"
echo "Installing Python libraries..."
echo "========================================"

python3 -m pip install --upgrade pip

python3 -m pip install \
    numpy \
    pandas \
    pyarrow \
    boto3 \
    botocore \
    requests \
    numba

echo "========================================"
echo "Downloading scripts..."
echo "========================================"

download() {
    local url="$1"
    local output="$2"

    for i in {1..10}; do
        echo "Downloading $output (Attempt $i/10)..."

        if wget -O "$output" "$url"; then
            echo "$output downloaded successfully."
            return 0
        fi

        echo "Download failed. Retrying in 5 seconds..."
        sleep 5
    done

    echo "ERROR: Failed to download $output after 10 attempts."
    exit 1
}

download \
"https://raw.githubusercontent.com/GitHubExodus/stocks/main/indicators_REAL_FAST.py" \
"indicators_REAL_FAST.py"

download \
"https://raw.githubusercontent.com/GitHubExodus/stocks/main/run_loop.py" \
"run_loop.py"

echo "========================================"
echo "Starting run_loop.py..."
echo "========================================"

nohup bash -c '
while true
do
    python3 run_loop.py
    echo "Program exited. Restarting in 5 seconds..."
    sleep 5
done
' > run.log 2>&1 &

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo "Process ID: $!"
echo ""
echo "Useful commands:"
echo "  View live logs:"
echo "      tail -f run.log"
echo ""
echo "  Check if running:"
echo "      ps -ef | grep run_loop.py"
echo ""
echo "  Stop the program:"
echo "      pkill -f run_loop.py"
echo ""
echo "You can now safely close your browser."
