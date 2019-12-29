#!/bin/bash
target="/home/garrett/Documents/Github/codats-paper/figures"
for file in result_plots/multisource_average*.pdf; do
    cp -a "$file" "$target/$(basename "${file// /_}")"
done
