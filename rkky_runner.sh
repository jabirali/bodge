#!/usr/bin/env bash

for sep in 2 5 10; do
	for winding in $(seq 0 10); do
		for s1 in "x+" "y+" "x-" "y-"; do
			for s2 in "x+" "y+" "x-" "y-"; do
				./rkky_sp_open.py "$sep" "$s1" "$s2" 0.10 0.10 --winding="$winding" --filename="OUTPUT.csv"
				./rkky_sp_pbc.py  "$sep" "$s1" "$s2" 0.10 0.10 --winding="$winding" --filename="OUTPUT.csv"
			done
		done
	done
done
