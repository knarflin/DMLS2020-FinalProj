#!/bin/bash

path_prefix="./splitdata_small/train/"
mkdir ${path_prefix}total
cp -r ${path_prefix}site0/* ${path_prefix}total/

for counter in $(seq 1 3)
do
    for site_counter in $(seq -f "%05g" 0 42)
    do
        DIR="${path_prefix}site${counter}/${site_counter}"
        for f in "$DIR"/*
        do
            cp $f ${path_prefix}total/${site_counter}/
        done
    done
done
