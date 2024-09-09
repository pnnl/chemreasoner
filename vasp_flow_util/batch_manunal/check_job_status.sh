#!/bin/bash

n1=0
n2=0
n3=0
n4=0
n5=0
n6=0

for j in job_1*; do  # loop all jobs 
  #echo $j
  cd $j

    if [ ! -f 'job.log' ]; then # If job.log is absent, the job was not started
      #echo $j ' Not started'
      n1=$(($n1 + 1))

    #elif [ ! -f 'slurm.out' ]; then
    elif [ $(ls slurm-*.out | wc -l) = 0 ]; then
      echo $j 'No slurm written'
      nb2=$(($nb2 + 1))

    else  # otherwise, it started
      match=$(grep -c "reached required accuracy" job.log)
      slurm_out=$(ls -altr slurm-*.out | tail -1 | awk '{ print $NF }')  

      if [ $match = 1 ]; then  # Job started, and finished for convergence
        n3=$(($n3 + 1))
        #echo $j 'Job done with CONTCAR. Energy: ' $e

      else ## Otherwise, the job was not done properly and not finished
        if [ ! -f 'CONTCAR' ]; then ## If so, we need to continue this run?
          echo $j 'Started but something went wrong. No CONTCAR written'
          n4=$(($n4 + 1))
        else  ## This is either running or terminated earlier or force stop (e.g. 10 step max)
          nlines=$( grep E0 job.log | wc -l)
	  if (( $nlines > 9 )); then 
		  #job_status='Stopped due to step max or convergence'
       	          n5=$(($n5 + 1))
          else
		  job_status='Running or stopped earlier at: '$nlines
		  n6=$(($n6 + 1))
                  echo $j 'Started but not finished. '  $job_status
                  more $slurm_out
                  #sbatch ../../run_vasp.sh
	  fi	 
          #echo $j 'Started but not finished.'  $job_status
        fi
      fi
    fi
  cd ../
done

echo 'Not started:' $n1 'No Slurm:' $n2 'Terminated max:' $n5 'Running/stop no-max:' $n6 'No CONTCAR:' $n4 'Finished:' $n3

