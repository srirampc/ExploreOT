#!/bin/bash
for num in 20 50 100 200
do
    for graph in "gaussian_graph" "gaussian_point_graph"
    do
        for loss in "1_loss" "square_loss"
        do
            jobname="${graph}_${num}_${num}_${loss}___2" 
            filename="${num}_${num}_${loss}___2.pickle"
            filepath="./pickle_compare/${graph}/gromov/${filename}"
            if [ ! -e $filepath ]
            then
                echo $filepath
                sbatch -J $jobname --mem-per-cpu=2G gromov.sbatch $num $loss $graph               
            fi                
        done
    done
done

#!/bin/bash
for num in 500 1000 1001
do
    for loss in "1_loss" "square_loss"
    do
        jobname="${graph}_${num}_${num}_${loss}___2" 
        filename="${num}_${num}_${loss}___2.pickle"
        filepath="./pickle_compare/gaussian_point_graph/gromov/${filename}"
        if [ ! -e $filepath ]
        then
            echo $filepath
            sbatch -J $jobname --mem-per-cpu=2G gromov.sbatch $num $loss "gaussian_point_graph"               
        fi             
    done
done