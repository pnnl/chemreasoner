#!/bin/bash

max_num_submitted=50
n=0

for j in job_a*; do 
  echo $j
  cd $j
    if [ ! -f 'job.log' ]; then
        sbatch ../../run_vasp.sh
	n=$((n+1))
	echo $n
    fi

    if [ $n -eq $max_num_submitted ]; then
	    cd ../
	    break
    fi

  cd ../
done

