#!/bin/bash
target="/home/garrett/Documents/Github/multi-source-adaptation-paper/figures"
for file in result_plots/multisource_{method,}average*.pdf result_plots/varyamount_{method,}average*.pdf; do
    cp -a "$file" "$target/$(basename "${file// /_}")"
done
