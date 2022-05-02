#!/bin/bash

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/lib
export CUDA_VISIBLE_DEVICES="0"

echo ""
echo -n "input the port:"
read port

export HOME=/home/zhaoshuai
logdir="${HOME}/models/eclip"


# sleep time, hours
sleep_t=6
times=0

# while loop
# To aviod memory leak, run the tensorboard command periodly
while true
do
	# https://stackoverflow.com/questions/40106949/unable-to-open-tensorboard-in-browser
	tensorboard --bind_all --logdir=${logdir} --port=${port} &
	#tensorboard --logdir=${logdir} --port=${port} &
	last_pid=$!
	sleep ${sleep_t}h
	kill -9 ${last_pid}
	times=`expr ${times} + 1`
	echo "Restart tensorboard ${times} times."
done

echo "tensorboard is stopped!"
