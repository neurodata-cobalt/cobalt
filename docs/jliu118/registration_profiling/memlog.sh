## memlog.sh
#!/bin/bash -e
# run this in the background with nohup ./memlog.sh > mem.txt &
#
while true; do
    echo "$(free -m | grep buffers/cache | awk '{print $3}')"
    sleep 1
done