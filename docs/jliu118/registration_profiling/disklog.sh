#!/bin/bash -e
# run this in the background with nohup ./disklog.sh > disk.txt &
#
while true; do
    echo "$(du -s $1 | awk '{print $1}')"
    sleep 30
done
