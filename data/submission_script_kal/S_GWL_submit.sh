#!/bin/bash
for num in 1001 20 50 100 200 500 1000 1001
do
    for epsilon in 0.001 0.01 0.1 1 2 10 100
    do
        for graph in "gaussian_graph" "gaussian_point_graph"
        do
            for loss in "1_loss" "square_loss"
            do
                jobname="${graph}_${num}_${num}_${loss}_${epsilon}__2" 
                filename="${num}_${num}_${loss}_${epsilon}__2.pickle"
                filepath="./pickle_compare/${graph}/S_GWL/${filename}"
                if [ ! -e $filepath ]
                then
                    echo $filepath
                    sbatch -J $jobname --mem-per-cpu=2G S_GWL.sbatch $num $loss $graph $epsilon                
                fi
            done
        done
    done
done

for epsilon in 0.001 0.01 0.1 1 2 10 100
do
    for loss in "1_loss" "square_loss"
    do
        jobname="gaussian_point_graph_1500_1500_${loss}_${epsilon}__2" 
        filename="1500_1500_${loss}_${epsilon}__2.pickle"
        filepath="./pickle_compare/gaussian_point_graph/S_GWL/${filename}"
        if [ ! -e $filepath ]
        then
            echo $filepath
            sbatch -J $jobname --mem-per-cpu=2G S_GWL.sbatch 1500 $loss gaussian_point_graph $epsilon                
        fi         
    done
done
