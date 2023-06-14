#!/bin/bash
for num in 20 50 100 200 500 1000 1001
do
    for loss in "1_loss" "square_loss"
    do
        jobname="gaussian_graph_${num}_${num}_${loss}___2" 
        filename="${num}_${num}_${loss}___2.pickle"
        filepath="./pickle_compare/gaussian_graph/uniform/${filename}"
        if [ ! -e $filepath ]
        then
            echo $filepath
            sbatch -J $jobname --mem-per-cpu=2G uniform.sbatch $num $loss "gaussian_graph"                
        fi
    done
done