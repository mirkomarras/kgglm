
NPROC=2
for hops in 3 5;
  do
  	for npaths in  250 500 1000 1500 2000 2500 3000 10000 ;
			do
	  		for dataset in ml1m lfm1m;
					do			
						echo 'Creating: dataset-' $dataset ' npaths-' $npaths ' hops-' $hops
						bash create_dataset.sh $dataset $npaths $hops $NPROC
						echo 'Completed'
						echo 
					done
			done    	
  done
