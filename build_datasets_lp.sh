#!/bin/bash


NPROC=4
for hops in 1;#3 5;
  do
  	for npaths in 250; 
			do
	  		for dataset in ml1m lfm1m;
					do			
						echo 'Creating: dataset-' $dataset ' npaths-' $npaths ' hops-' $hops
						bash create_dataset_lp.sh $dataset $npaths $hops $NPROC
						echo 'Completed'
						echo 
					done
			done    	
  done
