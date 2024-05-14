#!/usr/bin/env bash

for sep in 5 2 10; do
	for potential in $(seq -4.0 0.5 4.0); do
		for s1 in "x+" "y+" "x-" "y-"; do
			for s2 in "x+" "y+" "x-" "y-"; do
				./rkky_sp.py "$sep" "$s1" "$s2" 0.10 0.10 --filename="OUTPUT_μ.csv" --cuda
				./rkky_sp.py "$sep" "$s1" "$s2" 0.10 0.10 --filename="OUTPUT_μ.csv" --cuda
			done
		done
	done
done

# TODO: Fermi level?