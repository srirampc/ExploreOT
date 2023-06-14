#!/bin/bash
for num in 20 50 100 200 500 1000 1001 1500
do
    for loss in "1_loss" "square_loss"
    do
        jobname="gaussian_point_graph_${num}_${num}_${loss}___2" 
        filename="${num}_${num}_${loss}___2.pickle"
        filepath="./pickle_compare/gaussian_point_graph/sliced_gromov/${filename}"
        if [ ! -e $filepath ]
        then
            echo $filepath
            sbatch -J $jobname --mem-per-cpu=2G sliced_gromov.sbatch $num $loss "gaussian_point_graph"               
        fi
    done
done

for num in 2000 5000 10000
do
    for loss in "1_loss" "square_loss"
    do
        jobname="gaussian_point_graph_${num}_${num}_${loss}___2" 
        filename="${num}_${num}_${loss}___2.pickle"
        filepath="./pickle_compare/gaussian_point_graph/sliced_gromov/${filename}"
        if [ ! -e $filepath ]
        then
            echo $filepath
            sbatch -J $jobname --mem-per-cpu=4G sliced_gromov.sbatch $num $loss "gaussian_point_graph"               
        fi
    done
done