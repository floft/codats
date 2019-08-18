#!/bin/bash
#
# Get how long each 15k training took for each method
# We'll compute mean +/- std in Python with analyze_time.py
#
output="training_time.txt"

echo "Outputting to $output"

dataset="freqscaleshiftrotate"
for method in vrada rdann rdann cycada random dann deepjdot dann_grl none upper; do
	sacct -S 0812 -E 0815 | grep synthetic1 | tr -s ' ' | cut -d' ' -f1,6 | while read -r filename duration; do
		out="$(sed 's/$/.out/g' <<< "$filename" | sed 's/^/slurm_logs\/train_/g')"
		err="$(sed 's/$/.err/g' <<< "$filename" | sed 's/^/slurm_logs\/train_/g')"
		head="$(head -n 5 "$out")"

		# Check that we didn't load from checkpoint, then check if it's the right dataset and method
		if grep "step 15001" "$err" &>/dev/null \
			&& grep "$dataset" <<< "$head" &>/dev/null \
			&& grep "$method" <<< "$head" &>/dev/null; then
			echo "$method $duration"
		fi
	done
done | tee "$output"
