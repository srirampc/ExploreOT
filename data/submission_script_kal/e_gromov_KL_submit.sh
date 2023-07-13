#!/bin/bash
for num in 20 50 100 200
do
    for epsilon in 0.001 0.005 0.01 0.05 0.1 1
    do
        for graph in "gaussian_graph" "gaussian_point_graph"
        do
            for loss in "1_loss" "square_loss"
            do
                jobname="${graph}_${num}_${num}_${loss}_${epsilon}__2" 
                filename="${num}_${num}_${loss}_${epsilon}__2.pickle"
                filepath="./pickle_compare/${graph}/e_gromov_KL/${filename}"
                if [ ! -e $filepath ]
                then
                    echo $filepath
                    sbatch -J $jobname --mem-per-cpu=2G e_gromov_KL.sbatch $num $loss $graph $epsilon                
                fi                
            done
        done
    done
done
