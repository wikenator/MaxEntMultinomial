#!/bin/bash

counter=1

for f in algebra_*.txt
do
	while read l; do
		echo "problem: $counter"
		echo "$counter#@#$l" >> "algebra.txt"

		counter=$((counter+1))
	done < "$f"

	#mv "$f.bak" "$f"
done

for f in geometry_*.txt
do
	while read l; do
		echo "problem: $counter"
		echo "$counter#@#$l" >> "geometry.txt"

		counter=$((counter+1))
	done < "$f"

	#mv "$f.bak" "$f"
done

for f in arithmetic_*.txt
do
	while read l; do
		echo "problem: $counter"
		echo "$counter#@#$l" >> "arithmetic.txt"

		counter=$((counter+1))
	done < "$f"

	#mv "$f.bak" "$f"
done
