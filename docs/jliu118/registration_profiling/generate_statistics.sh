## generate_statistics.sh
# A function for generating memory and cpu summaries for 50 um ARA registration with ndreg of s3617.
#
# Usage: ./generate_statistics.sh /path/to/output/directory

rm -rf $1
mkdir $1

./memlog.sh > ${1}/mem.txt &
memkey=$!
python cpulog.py ${1}/cpu.txt &
cpukey=$!
./disklog.sh $1 > ${1}/disk.txt &
diskkey=$!

python reg_s3617_50um.py

kill $memkey $cpukey $diskkey
