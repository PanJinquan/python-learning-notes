#!/usr/bin/env bash
echo "build..."
mkdir build
cd build
cmake ..
make -j4

sleep 1
echo "Run..."
cd ../
./bin/Demo

sleep 1
rm -rf build