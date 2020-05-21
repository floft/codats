#!/bin/bash
target="/home/garrett/Documents/Github/codats-paper/figures"
for file in result_plots_paper/*_accuracy.pdf; do
    cp -a "$file" "$target/$(basename "${file// /_}")"
done

cp class_balance_*.pdf "$target/"
