#!/bin/bash

set -e

echo "Updating Ubuntu..."
apt update
apt install -y python3 python3-pip wget

echo "Installing Python libraries..."
python3 -m pip install --upgrade pip
python3 -m pip install \
    numpy \
    pandas \
    pyarrow \
    boto3 \
    botocore \
    requests \
    numba

echo "Downloading scripts..."

wget -O indicators_REAL_FAST.py https://raw.githubusercontent.com/GitHubExodus/stocks/main/indicators_REAL_FAST.py
wget -O run_loop.py https://raw.githubusercontent.com/GitHubExodus/stocks/main/run_loop.py

echo "Starting..."

nohup python3 run_loop.py > run.log 2>&1 &

echo "Done!"
echo "PID: $!"
echo "View logs with:"
echo "tail -f run.log"
