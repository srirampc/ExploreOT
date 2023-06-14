#!/bin/bash
for num in 20 50 100 200 500 1000
do
    for batchsize in 1 10 100 1000
    do
        for graph in "gaussian_graph" "gaussian_point_graph"
        do
            for iter in 10 50 100 500 1000
            do
                for loss in "1_loss" "square_loss"
                do
                    jobname="${graph}_${num}_${num}_${loss}_${batchsize}_${iter}_2" 
                    filename="${num}_${num}_${loss}_${batchsize}_${iter}_2.pickle"
                    filepath="./pickle_compare/${graph}/sampled_gromov_e_constant/${filename}"
                    if [ ! -e $filepath ]
                    then
                        echo $filepath
                        sbatch -J $jobname --mem-per-cpu=2G sampled_gromov_e_constant.sbatch $num $loss $graph $batchsize $iter
                    fi
                done
            done
        done
    done
done

for num in 2000 5000 10000
do
    for batchsize in 1 10 100 1000
    do
        for graph in "gaussian_graph" "gaussian_point_graph"
        do
            for iter in 10 50 100 500 1000
            do
                for loss in "1_loss" "square_loss"
                do
                    jobname="${graph}_${num}_${num}_${loss}_${batchsize}_${iter}_2" 
                    filename="${num}_${num}_${loss}_${batchsize}_${iter}_2.pickle"
                    filepath="./pickle_compare/${graph}/sampled_gromov_e_constant/${filename}"
                    if [ ! -e $filepath ]
                    then
                        echo $filepath
                        sbatch -J $jobname --mem-per-cpu=4G sampled_gromov_e_constant.sbatch $num $loss $graph $batchsize $iter
                    fi
                done
            done
        done
    done
done