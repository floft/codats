#!/bin/bash
target="/home/garrett/Documents/Github/multi-source-adaptation-paper/figures"
for file in results/multisource_methodaverage*.pdf results/multisource_average*.pdf; do
    cp -a "$file" "$target/$(basename "${file// /_}")"
done
