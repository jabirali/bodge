#!/usr/bin/env bash

#for potential in $(seq -3.95 0.05 0.00); do
for potential in -4.15 -4.20 -4.25; do
    for sep in 2 3 4 5; do
        for s1 in "x+" "y+" "x-" "y-"; do
            for s2 in "x+" "y+" "x-" "y-"; do
                ./rkky_sp.py "$sep" "$s1" "$s2" --potential="$potential" --filename="OUTPUT_μ.csv" --cuda
            done
        done
    done
done

# TODO: Fermi level?
