#!/bin/bash


NPROC=4
for hops in 5 7 ;#3 5;
  do
  	for npaths in 500; #250 500 1000 1500 2000 2500 3000 10000 ;
			do
	  		for dataset in hummus; # ml1m lfm1m;
					do			
						echo 'Creating: dataset-' $dataset ' npaths-' $npaths ' hops-' $hops
						bash create_dataset_rec.sh $dataset $npaths $hops $NPROC
						echo 'Completed'
						echo 
					done
			done    	
  done
