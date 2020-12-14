#!/bin/sh

# Try 10 random searches
for i in $(seq 0 1 9)
  do
    # Limit runtime in case of no convergence
    echo "Trying random combination $((i + 1))"
    timeout 1800s python fine_tune_arimax.py
done 2>&1 | tee "./models/params/arimax_ft_log.txt"
