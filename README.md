<!-- vscode-markdown-toc -->


<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


# Unified Causal Language Modeling for Recommendation and Link Prediction on Knowledge Graphs

<!--This Repository contain the code for the paper "Knowledge Graph Embeddings as Masked Language Models: Link Prediction, Recommendation, Explainability".-->


This work it's based on the evaluation functions and code style of the paper "PEARLM: Faithful Path-Based Explainable Recommendation via Language Modeling over Knowledge Graphs [24]".

**Note:** all experiments have been run with fixed seed in order to ease reproducibility of the results. In the data folder you can find the datasets for link prediction from the RotatE paper [13].



- [Unified Casual Language Modeling for Recommendation and Link Prediction on Knowledge Graphs](#unified-casual-language-modeling-for-recommendation-and-link-prediction-on-knowledge-graphs)
	- [Requirements](#requirements)
	- [Usage](#usage)
		- [Run the Experiments](#run-the-experiments)
		- [Reproducibility](#reproducibility)
	- [Datasets](#datasets)
	- [Results](#results)
		- [Recommendation](#recommendation)
		- [Link Prediction](#link-prediction)
		- [Training and Evaluation Time on NVIDIA Tesla P6](#training-and-evaluation-time-on-nvidia-tesla-p6)
		- [Training and Evaluation Time on NVIDIA RTX A6000](#training-and-evaluation-time-on-nvidia-rtx-a6000)
	- [References](#references)
	- [Contributing](#contributing)
	- [Citations](#citations)
	- [License](#license)


## <a name='Requirements'></a>Requirements
- Python 3.9.13


It is possible to download the weights and paths for UKGCLM from https://bit.ly/4aEWF0Z

*Make sure to download the `data/` folder present in onedrive and copy it into the `fix/` folder before proceeding.*

## <a name='Usage'></a>Usage
**Steps.**
1. Clone of the repository by running `git clone --recurse-submodules https://github.com/mirkomarras/lm-vs-gnn-embeddings.git`
2. Execute `chmod +x fix/fix.sh && ./fix/fix.sh`
3. Install the required packages with `pip install -r requirements.txt`
4. Install PEARLM with `pip install pearlm/.`
5. Run `CUDA_DEVICE=0 && ./run_kge_experiments_recommendation.sh CUDA_DEVICE` to train and view the results for the baselines on Recommendation
6. Run `CUDA_DEVICE=0 && ./run_kge_evaluation_recommendation.sh CUDA_DEVICE` to view the results on the best embeddings for the baselines on Recommendation
7. Run `CUDA_DEVICE=0 && ./run_kge_experiments_linkprediction.sh CUDA_DEVICE` to train and view the results for the baselines on Link Prediction
8. Run `CUDA_DEVICE=0 && ./run_kge_evaluation_linkprediction.sh CUDA_DEVICE` to view the results on the best embeddings for the baselines on Link Prediction
9. Run `CUDA_DEVICE=0 && ./run_ukgclm_experiments.sh CUDA_DEVICE` to view the results on the best embeddings for UKGCLM on Recommendation and Link Prediction



### <a name='RuntheExperiments'></a>Run the Experiments
To start the training for the reproducibility of recommendation results (the same is for link prediction), in background, from the root folder, run:

```sh
> CUDA_DEVICE=0

> screen -S baseline_computation -L -Logfile baseline.txt ./run_kge_experiments_recommendation.sh CUDA_DEVICE 
```
To detach from the terminal and let the server reproduce the experiments, press `CTRL+A` and press `d` (You can disconnect from the server and the training doesn't stop)

To go back to the screen session, type `screen -r`


Inside the bash script it's possible to modify the hyperparameters of a model, all the models are executed sequentially as it appears here:

```sh
### TransE: ml1m ###
start=$(date +%s) # Start time to view at the end the total time of run
echo -e "\n\t\t$(date)\n" # view the full date of start
model="TransE"
dataset="ml1m" 
emb_size="100" # Embedding Size
batch_size="64" # Batch Size
lr="0.0001" # Learning Rate
wd="0" # Weight Decay
k="10" # Top-k
use_cuda="all" 
margin=1 # For the loss function
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n" # An header that will be displayed at the start of the execution
# We set the CUDA_VISIBLE_DEVICE to set which gpu to use and we'll run the training in background. After, we save the corresponding time of execution in the file "elapsed_training.txt" by appending it.
export CUDA_VISIBLE_DEVICES=$GPU && python pearlm/pathlm/models/kge_rec/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_$dataset.txt && end=$(date +%s) && runtime=$((end-start)) && echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds\n" >> elapsed_training.txt
```

### <a name='Reproducibility'></a>Reproducibility 

The reproducibility of the experiments can be done by running `./run_kge_evaluations_recommendation.sh CUDA_DEVICE` or `./run_kge_evaluations_linkprediction.sh CUDA_DEVICE` where the models are executed sequentially with the default embeddings the ones that results to the best performance on the datasets.

First of all it is necessary to download and copy the folders "Best Checkpoint for Reproducibility" and "UKGCLM Weights" from the link above (https://bit.ly/4aEWF0Z) to download the best weights, then you can run `./setup_reproducibility.sh` to automatically copy the weights in the correct folders (after you've downloaded the folder and moved it on the root). Finally, execute `./run_kge_evaluations_recommendation.sh CUDA_DEVICE` or `./run_ukgclm_experiments.sh CUDA_DEVICE` to test the models on their best weights. 

By executing it, you will obtain the same results as in the paper.

For Example, for TransE on ml1m and lfm1m for Recommendation:
```sh
# On ml1m
python pearlm/pathlm/models/kge_rec/TransE/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TransE_dataset_ml1m_ndcg_0.28_mrr_0.23_prec_0.1_rec_0.03_ser_0.33_div_0.41_nov_0.93_cov_0.04_epoch_16_e100_bs64_lr0.0001.pth

# On lfm1m
python pearlm/pathlm/models/kge_rec/TransE/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TransE_dataset_lfm1m_ndcg_0.12_mrr_0.1_prec_0.03_rec_0.01_ser_0.59_div_0.54_nov_0.85_cov_0.0_epoch_28_e100_bs64_lr0.001.pth

```
>*Remember to change the embed_size parameter when you change the checkpoint for a model. In general, to test other weights, you have to move the weight ".pth" inside pearlm/pathlm/models/kge_rec/TransE/weight_dir_ckpt if the model to test is TransE for example. You can evaluate on link prediction task by changing the checkpoint and setting --lp True*


## <a name='Datasets'></a>Datasets

From [24]
|                             | ML1M        | LFM1M      |
|-----------------------------|-------------|------------|
| **Interaction Information** |             |            |
| Users                       | 6,040        | 4,817       |
| Products                    | 2,984        | 12,492      |
| Interactions                | 932,295      | 1,091,275    |
| Density                     | 0.05        | 0.01       |
| **Knowledge Information**   |             |            |
| Entities (Types)            | 13,804 (12)  | 17,492 (5)  |
| Relations (Types)           | 193,089 (11) | 219,084 (4) |
| Sparsity                    | 0.0060      | 0.0035     |
| Avg. Degree Overall         | 28.07       | 25.05      |
| Avg. Degree Products        | 64.86       | 17.53      |


|                             | FB15k-237   | WN18RR     |
|-----------------------------|-------------|------------|
| **Dataset Information**     |             |            |
| Entities                    | 6,040        | 89,869       |
| Relations                   | 2,984        | 12,492      |
| #Training                   | 289,650        | 12,492      |
| #Test                   | 20,446        | 3,134      |

## <a name='Results'></a>Results
### Recommendation
**ML1M**
|          | NDCG | MRR  | Precision | Recall | Serendipity | Diversity | Novelty | Converage | Parameters: [Epoch, EmbSize, BatchSize, lr] |
|----------|------|------|-----------|--------|-------------|-----------|---------|-----------|---------------------------------------------|
| TransE [1]  | 0.28 |  0.23  |     0.10   |  0.03  |   0.33    |     0.41     |    0.93  |    0.04     |         [16,100,64,0.0001]            |
| TransH [2]   |  0.26 | 0.20 |    0.09   | 0.03  |    0.43   |   0.45   |   0.93  |    0.14|        [10,100,64,0.0001]              |
| TransD [3]  |  0.23  |  0.18  | 0.08      |   0.03  |     0.84     |    0.42    |  0.94    |     0.48   |        [27,100,256,0.0001]          |
| TransR [4]  |  0.16  |  0.12  |     0.05    | 0.02   |  0.85         |   0.47     |  0.93  |    0.95  |       [2,100,256,0.001]           |
| TorusE [5] |  0.26  |   0.21  |   0.08    |   0.03  | 0.61        |    0.43    |   0.88    |  0.04      |        [0,100,64,0.0001]                    |
| RotatE [13] |  0.20  |  0.15  |   0.07      |     0.02 |    0.76      |   0.49      |   0.93   |   0.40     |        [23,100,256,0.0001]       |
| RESCAL [6] |  0.26  |  0.20  |   0.09     |    0.03  |    0.42   | 0.41      |  0.92    |    0.19 |         [3,100,64,0.0001]               |
| DistMult [7]|  0.28  |  0.22  |    0.11   |   0.04  |   0.32    |   0.40      | 0.93    |    0.07    |        [22,100,64,0.0001]                  |
| ComplEX [8]| 0.26   | 0.21   |   0.10       |  0.03    |     0.30     |      0.37  |   0.93    | 0.05      |  [2,100,64,0.0001]  | 
| TuckER [14] |  0.27  | 0.22   |     0.10   |   0.04  |    0.16       |  0.36       |  0.93    |    0.04    |         [19,100,256,0.0001]                 |
| Analogy [9] | 0.26  |  0.21   | 0.10         |  0.03     |        0.29    |    0.33      |    0.93    |    0.06      |[4,100,64,0.0001]         |
| HolE  [12]  | 0.24  |  0.19  |   0.09     | 0.03     |    0.50     |   0.39      |    0.93  |    0.14   |      [11,100,64,0.0001]                |
| ConvKB [10] |   -  |   -  |        -  |     -  |       -    |       -   |      -  |          -| -                      |
| ConvE [11]  |  0.27  |  0.22  |   0.10      | 0.04     |   0.22        |      0.38  |    0.93  |  0.04     |       [2,200,256,0.0001]                 |
| CKE [17]  |  0.30  |  0.23  |   0.11      | 0.04     |   0.67  |     0.38 |    0.92  |  0.30    |         |
| CFKG [18]  |  0.27  |  0.21  |   0.10     | 0.03    |   0.22?        |   0.41    |  0.92   |  0.03     |         |
| KGAT [19]  |  0.31  |  0.24  |   0.11      | 0.04     |   ?0.49        |     0.40  |    ?0.92  |   0.13   |         |
| PGPR [20]  |  0.28 |  0.21  |   0.08      | 0.03     |       0.77   |  0.43     | 0.92    | 0.43     |         |
| UCPR [21]  |  0.26  |  0.19  |   0.07      | 0.03     |     0.78     |   0.42    |  0.93   | 0.52     |         |
| CAFE [22]  |  0.21  |  0.15  |   0.06      | 0.03     |       0.77   |   0.45    |    0.93 |   0.26   |         |
| PLM [23]  |  0.27  |  0.18  |   0.07      | 0.03    |      0.90    | 0.45      |   0.93  |  0.64     |         |
| PEARLM [24]  |  0.44  |  0.31  |   0.13      | 0.08    |    0.93     |   0.45    |   0.93  |  ?0.80     |         |
| UKGCLM gen-only(#EP.3)   |  0.11  |  0.07 |   0.03     | 0.01     |     0.98     |      ?0.54  |    0.93 |  0.74    |         |
| UKGCLM gen+spec(#EP.2)  |  0.41  |  0.31  |   0.14      | 0.08    |     0.94    |     0.47  |  0.93   |     0.79 |         |
| UKGCLM spec-only(#EP.20)  |  0.46 |  0.36 |   0.16      | 0.09    |        0.91  |     0.46  |    0.93 |    0.81  |         |






**LFM1M**


|          | NDCG | MRR  | Precision | Recall | Serendipity | Diversity | Novelty | Converage | Parameters: [Epoch, EmbSize, BatchSize, lr] |
|----------|------|------|-----------|--------|-------------|-----------|---------|-----------|---------------------------------------------|
| TransE [1] | 0.12 | 0.10   |   0.03      |  0.01     |   0.59    |  0.54    |  0.85    | 0.00     |    [28,100,64,0.001]                  |
| TransH [2] |   0.18 | 0.14 |    0.05   |  0.01   |    0.99       |  0.66       |  0.88     |    0.36     |     [22,100,64,0.001]                    |
| TransD [3] |  0.17 |  0.13  | 0.06      |   0.01   |  0.83     |    0.55  |    0.87  |   0.30    |         [29,100,256,0.0001]               |
| TransR [4] |   0.15 |  0.11  |    0.04    | 0.01   |       1.0    |  0.70       |  0.90     |   0.70      |        [29,100,256,0.001]                  |
| TorusE [5] | 0.18   |  0.14  |   0.05   |    0.01   | 0.90        |   0.62    |   0.86   |    0.30  |      [28,100,256,0.0001]            |
| RotatE [13] | 0.25  |  0.20 |      0.09 |   0.02  |     0.97|    0.52  |    0.87|     0.42   |      [29,100,256,0.001]            |
| RESCAL [6] | 0.24  | 0.19    |  0.08     |   0.02 |  0.99    | 0.61      |   0.88  |   0.54   |       [26,100,256,0.001]                     |
| DistMult [7] | 0.30  |   0.25  |     0.11   |   0.03 |       0.97   |    0.56  | 0.87    |  0.35   |       [10,100,256,0.001]                |
| ComplEX  [8]|   0.28 | 0.23    |      0.10  |    0.03  |   0.97      |    0.52   |    0.87   | 0.43      |           [29,100,64,0.0001]   |
| TuckER [14] |   0.12 |  0.10  |    0.03    |   0.01  |     0.32 |     0.29  |   0.83  |   0.02   |          [4,64,64,0.0001]              |
| Analogy [9] |  0.33   |  0.28  |   0.12     |   0.03   |    0.98     |   0.48       |     0.87   |  0.45   | [27,100,128,0.0001]                                   |
| HolE  [12]  |  0.19  | 0.14   |  0.06     |   0.02 |   0.96  |    0.55   |  0.87   |   0.27     |             [29,100,64,0.0001]                 |
| ConvKB [10] |   -  | -    | -          | -       |      -      |     -     |     -   |     -     | -                       |
| ConvE [11]  |  0.13  |  0.11   |   0.03      |    0.01  |        0.47   |    0.58     |    0.88   |  0.00       |   [2,200,64,0.0001]            |
| CKE [17]  |  0.33  |  0.27  |   0.12      | 0.03     |   0.92       |      0.48  |    0.86 |  0.23    |        |
| CFKG [18]  |  0.13  |  0.10  |   0.03      | 0.01     |   0.54       |      0.39 |    0.84  |  0.00   |                   |
| KGAT [19]  |  0.30  |  0.24  |   0.11      | 0.01     |   0.69       |      0.19  |   0.87  |  0.49   |                  |
| PGPR [20]  |  0.18 |  0.14  |   0.04  | 0.01   |   0.84        |      0.62 |    0.81  |  0.14     |                  |
| UCPR [21]  |  0.32 |  0.26  |   0.10      | 0.03    |   0.98        |      0.56  |    0.87  |  0.40     |               |
| CAFE [22]  |  0.14  |  0.09  |   0.04      | 0.01    |   0.82        |      0.63  |    0.86  |  0.03     |               |
| PLM [23]  |  0.28  |  0.19  |   0.08      | 0.02   |   0.98       |      0.63  |    0.87 |  0.45    |               |
| PEARLM [24]  |  0.59  |  0.51 |  0.29     | 0.12   |   0.97        |      0.59 |    0.88  |  0.78     |               |
| UKGCLM gen-only (#EP.3)   |  0.27  |  0.20 |   0.09      | 0.03  |   1.00        |      0.48  |    0.90 |  0.39     |               |
| UKGCLM gen+spec (#EP.2)  |  0.53  |  0.45 |   0.23      | 0.10    |   0.98        |      0.52  |    0.88  |  0.74     |               |
| UKGCLM spec-only (#EP.20) |  0.57  |  0.49  |   0.26     | 0.11     |   0.98       |      0.60  |    0.88  |  0.77     |               |

### Link Prediction

**ml1m**
|          | NDCG | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Parameters: [Epoch, EmbSize, BatchSize, lr] |
|----------|------|----|-----|--------|--------|---------|---------------------------------------------|
| TransE [1]  |   0.22   |  1.00  |   14.45  |   21.30     |      31.74  |   30.22      |         [23, 100, 64, 0.0001, 0]               |
| TransH [2]  |     0.22 |   0.99 |  0.19   |    14.84    |  20.97      | 31.64        |         [24 ,100 ,64 ,0.0001, 0]             |
| TransD [3]  |    0.21  |  1.54  |   0.16  |  6.58      |      21.34  |     38.90    |         [29, 100, 64, 0.0001, 0]              |
| TransR [4] |     0.17 |   1.29 |   0.13  |  6.36      |  14.94      |      30.43   |          [29 ,64, 128, 0.01, 0]            |
| TorusE [5] |    0.22  |  1.04  |    0.20 |   15.06     |     21.40   |      32.52   |         [17, 100, 64, 0.0001, 0]               |
| RotatE [13] |   0.28   |   1.83 |  0.22   |    13.12   |  26.26     |   47.50      |        [12 ,64, 128, 0.01, 0]               |
| RESCAL [6]  |     0.22 |  0.81  |   0.20  |     16.34   |     20.95   |      29.16   |        [3, 100, 64, 0.0001, 0]              |
| DistMult [7] |  0.24    |  1.53  |  0.19   |   11.11     |  22.98      |    40.76     |      [29, 100, 64, 0.0001, 0]                  |
| ComplEX [8] |    0.28  |  1.57  |     0.23 |     14.68   |  26.94      |  44.42       |      [29, 100, 64, 0.0001, 0]                |
| TuckER [14] |    0.13  |  0.65   |   0.10  |      4.18  |     15.83   |  21.26       |        [26, 100, 256, 0.0001, 0]              |
| Analogy [9] |  0.25    |   1.55 |   0.20  |    12.57    |   23.85     |    41.43     |      [18, 100, 64, 0.0001, 0]              |
| HolE  [12]  |    0.17  |   0.94 | 0.14    |  9.78      |   16.09     |     26.56    |      [24, 100, 64, 0.0001, 0]                |
| ConvKB [10] |   -  |    |    - |       - |     -   |   -      |              -                              |
| ConvE [11]  |  0.13    |  0.69  |   0.10  | 4.18       | 15.78      |  21.64        |      [1 ,200 ,64, 0.0001, 0]                |
| UKGCLM gen-only (#EP.3)  |  0.42  |  1.59  |   0.38     | 28.79     |   43.75        |      58.90  |                  |
| UKGCLM gen+spec (#EP.2)  |  0.43  |  1.61 |   0.39   | 29.92   |   44.54      |      59.86  |             |

**lfm1m**
|          | NDCG | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Parameters: [Epoch, EmbSize, BatchSize, lr] |
|----------|------|----|-----|--------|--------|---------|---------------------------------------------|
| TransE [1]  | 0.79     |  2.0 |   0.76  |    66.19    |     82.55   |    97.64     |       [21, 100, 64, 0.0001, 0]             |
| TransH [2]  |   0.79   |   2.07 |   0.76  |   66.38     |   82.05     |    97.91     |           [21, 100, 64, 0.0001, 0]             |
| TransD [3]  | 0.52     |  3.36  |  0.31   | 0.27       |   55.45     |     92.09    |           [4, 100, 64, 0.0001, 0]               |
| TransR [4] |   0.45   |  2.93  | 0.35    |    18.96    |  43.66      |   77.36      |            [18 ,64, 128, 0.01, 0]            |
| TorusE [5] |    0.79  |  1.71  | 0.79    |    68.59    |  88.33     |    98.29     |          [7, 100, 64, 0.0001, 0]           |
| RotatE [13] |     0.11 |   1.57 |  0.07   |    1.67    |   6.67    |     26.33    |       [0, 64, 128, 0.01, 0]               |
| RESCAL [6]  |   0.33   |  1.30  |  0.29   |   21.25     | 33.62      | 46.26       |        [3, 100, 64, 0.0001, 0]               |
| DistMult [7] |   0.24   | 2.28   |    0.17 |    7.13    |     19.85   |     47.89    |       [14 ,100, 64, 0.0001, 0]               |
| ComplEX [8] |    0.69  |  1.79  |  0.68   |    57.77    |     75.07   |     87.59   |       [2, 100, 64, 0.0001, 0]                 |
| TuckER [14] |   0.79   |   1.45 |  0.76   |    66.34    |    83.21    |     91.43    |       [15, 100, 64, 0.0001, 0]                 |
| Analogy [9] |   0.54   |  1.84  |    0.50 |    37.26    | 60.88       |      74.95   |      [3, 100, 64, 0.0001, 0]              |
| HolE  [12]  |    0.38  |  2.13  |  0.33   | 22.80       |  36.22     |   60.45      |       [8, 100, 64, 0.0001, 0]             |
| ConvKB [10] |  -    | -   |   -  |  -      |     -   |   -     |         -                                    |
| ConvE [11]  |   0.79   |  1.45  | 0.76    |    67.00    |    83.21    |   91.51      |      [11,200 ,64, 0.0001, 0]                  |
| UKGCLM gen-only (#EP.3)  |  0.28  |  0.81 |   0.26      | 19.89     |   30.48        |     36.60  |                  |
| UKGCLM gen+spec (#EP.2) |  0.11  |  0.54 |   0.09    |4.30    |   11.79       |     17.14 |                   |

**Other Results for Link Prediction on our implemented baselines**

**FB15k-237**
|          | NDCG | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Parameters: [Epoch, EmbSize, BatchSize, lr, Weight Decay] |
|----------|------|----|-----|--------|--------|---------|---------------------------------------------|
| TransE [1]  |    0.46  |  1.66  |  0.41  |  31.83      |    46.37    |  62.52       |   [27, 100, 64, 0.001, 0]              |
| TransH [2]  |   0.46   |   1.59 |   0.41 |     31.59  |     47.60   |    62.00     |     [27, 100, 64, 0.001, 0]                   |
| TransD [3]  |   0.44  |   2.01 |  0.36   |    20.60   |  48.27    |    67.15     |     [29, 100, 64, 0.001, 0]                 |
| TransR [4]  |    0.27  |    1.85 |    0.19 |  2.80      |    31.63    |    50.11     |     [27, 100, 64, 0.001, 0]                 |
| TorusE [5]  |   0.42  |  1.63  |    0.37 |   27.26    |  42.17     |   58.02     |     [27, 100, 64, 0.001, 0]                   |
| RotatE  [13] |   0.49  |  1.79 |    0.43 |   32.16     |  50.39     |     67.20   |    [29 ,100 ,64, 0.001, 0]                   |
| RESCAL [6]  |    0.48  |  1.58 |   0.44 |  34.84    |    48.97   |     63.20   |    [29, 100, 256, 0.001, 0]                    |
| DistMult [7] |   0.42  |  1.61  |  0.37  |  26.71     |   42.92    |   58.28     |   [27, 100, 64, 0.0001, 0]                    |
| ComplEX [8] |    0.46 |   1.71 |  0.41  |    30.76    |   46.35    |    62.97    |   [26, 100, 64, 0.0001, 0]                    |
| TuckER [14] |  0.10   |  0.89  |   0.07  |   3.24     |    8.66    |    19.98    |    [0 ,100, 64, 0.001, 0]                |
| Analogy [9] |   0.47  |  1.78 |    0.41  |  31.25     |  47.19    |    64.67   |   [11, 100, 64, 0.001, 0]    |
| HolE [12]    |    0.41  |  1.73 |  0.35  |   24.13    |    41.80   |     58.85   |   [29, 100, 256, 0.001, 0]                   |
| ConvKB [10] |    - |    |  -  |   -    |     -   |       -  |       -                                     |
| ConvE  [11]  |   0.10  | 0.86  | 0.07   | 3.31      |     8.66  |   19.57     |   [12, 200, 64, 0.003, 0]                     |


**WN18RR**
|          | NDCG | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Parameters: [Epoch, EmbSize, BatchSize, lr] |
|----------|------|----|-----|--------|--------|---------|---------------------------------------------|
| TransE [1]  |   0.24   |   1.55 |  0.17   | 0.26      |     31.27   |     45.27    |        [17, 100, 64, 0.001, 0]                |
| TransH [2]  |   0.24   |   1.54  |   0.17  |  0.86      |    31.01    |   45.57      |        [18, 100, 64, 0.001, 0]             |
| TransD [3]  |    0.24  |  1.67  |  0.17   |   0.36     |     31.07   |    47.35     |       [29, 100, 64, 0.001, 0]               |
| TransR [4] |    0.13  |   0.76 |  0.09   |   0.0     |  17.67      |     23.53    |          [29, 100, 64, 0.001, 0]                |
| TorusE [5] |   0.21  |   1.32 |   0.15 |     0.99   |   26.17     |   38.62      |         [20, 100, 64, 0.001, 0]              |
| RESCAL [6]  |  0.34    |  0.72  |   0.32  |    28.99    |     34.94   |      39.58   |        [21, 100, 64, 0.001 ,0]                |
| DistMult [7] |  0.39   |   0.48 |   0.38  |     37.36   |    38.75    |  39.87       |      [11, 100, 64, 0.001, 0]                |
| ComplEX [8] |    0.43 |  0.58 |  0.42  |   41.07     |  42.79      |    44.84    |       [19, 100, 64, 0.001, 0]             |
| Analogy [9] |   0.44  |  0.64 | 0.43   |     42.22   |   43.91     |     46.6    |       [23, 100, 64, 0.001, 0]          |
| ConvKB [10] |    -  |  -  |  -   |-       |   -     |     -    |                   -                         |
| ConvE [11]  |   -  | -  | -   |  -     |     -  |  -      |          -                   |
| HolE  [12]  |   0.38  |   0.56 |   0.37  |   35.90     |     38.82   |     40.67    |      [29, 100, 64, 0.001, 0]                    |
| RotatE [13] |    0.41 | 0.64  |   0.40  |      38.75  |    40.64    |     43.71    |      [29, 100, 64, 0.001, 0]                 |
| TuckER [14] | -    |   - |  -  |    -   |  -      |   -      |    -                                         |

### <a name='TrainingandEvaluationTimeonNVIDIATeslaP6'></a>Training and Evaluation Time on NVIDIA Tesla P6

**Recommendation**
|          | ml1m                           | lfm1m                  |
|----------|--------------------------------|------------------------|
| TransE [1] | 45 minutes, 46 seconds | 1 hour, 1 minute, 31 seconds |
| TransH [2] | 44 minutes, 16 seconds | 1 hour, 8 minutes, 0 seconds|
| TransD [3] | 31 minutes, 47 seconds | 42 minutes, 57 seconds|
| TransR [4] | 19 minutes, 42 seconds | 45 minutes, 54 seconds|
| TorusE [5] | 21 minutes, 41 seconds | 32 minutes, 27 seconds|
| RotatE [13] | 27 minutes, 10 seconds | 37 minutes, 47 seconds |
| RESCAL [6] | 29 minutes, 39 seconds | 35 minutes, 55 seconds|
| DistMult [7]| 42 minutes, 26 seconds | 27 minutes, 51 seconds|
| ComplEx [8] | 30 minutes, 22 seconds | 1 hour, 6 minutes, 15 seconds|
| TuckER [14] | 27 minutes, 15 seconds | 30 minutes, 41 seconds|
| Analogy [9] | 32 minutes, 1 second | 44 minutes, 47 seconds |
| HolE   [12] | 1 hour, 34 minutes, 6 seconds | 2 hours, 2 minutes, 49 seconds |
| ConvKB  [10]| -                              | -                      |
| ConvE [11]  | 20 minutes, 33 seconds | 46 minutes, 35 seconds|

### Training and Evaluation Time on NVIDIA RTX A6000
**Link Prediction**

|          | FB15k-237                      | WN18RR                  |
|----------|--------------------------------|------------------------|
| TransE [1] | 17 minutes, 49 seconds | 4 minutes, 27 seconds |
| TransH [2] | 20 minutes, 57 seconds | 5 minutes, 22 seconds|
| TransD [3] | 25 minutes, 34 seconds | 9 minutes, 23 seconds|
| TransR [4] | 27 minutes, 25 seconds | 11 minutes, 23 seconds|
| TorusE [5] | 17 minutes, 33 seconds | 7 minutes, 12 seconds|
| RotatE [13] | 15 minutes, 39 seconds | 8 minutes, 30 seconds |
| RESCAL [6] | 14 minutes, 23 seconds | 6 minutes, 36 seconds|
| DistMult [7]| 17 minutes, 1 seconds | 6 minutes, 15 seconds|
| ComplEx [8] | 17 minutes, 34 seconds | 7 minutes, 39 seconds|
| TuckER [14] | 7 minutes, 17 seconds | - |
| Analogy [9] | 16 minutes, 36 seconds | 7 minutes, 10 seconds |
| HolE   [12] | 18 minutes, 52 seconds | 10 minutes, 55 seconds |
| ConvKB  [10]| -                              | -                      |
| ConvE [11]  | 10 minutes, 17 seconds | - |


|          | ml1m                           | lfm1m                  |
|----------|--------------------------------|------------------------|
| TransE [1] | 7 minutes, 27 seconds | 5 minute, 44 seconds |
| TransH [2] | 8 minutes, 50 seconds | 7 minutes, 13 seconds|
| TransD [3] | 12 minutes, 32 seconds | 5 minutes, 17 seconds|
| TransR [4] | 8 minutes, 37 seconds | 4 minutes, 4 seconds|
| TorusE [5] | 5 minutes, 29 seconds | 3 minutes, 47 seconds|
| RotatE [13] | 5 minutes, 0 seconds | 2 minutes, 27 seconds |
| RESCAL [6] | 5 minutes, 33 seconds | 3 minutes, 45 seconds|
| DistMult [7]| 8 minutes, 17 seconds | 5 minutes, 29 seconds|
| ComplEx [8] | 8 minutes, 49 seconds |  3 minutes, 42 seconds|
| TuckER [14] | 6 minutes, 37 seconds | 2 minutes, 55 seconds|
| Analogy [9] | 5 minutes, 53 second | 3 minutes, 35 seconds |
| HolE   [12] | 17 minutes, 53 seconds | 10 minutes, 44 seconds |
| ConvKB  [10]| -                              | -                      |
| ConvE [11]  | 5 minutes, 5 seconds | 3 minutes, 18 seconds|

## <a name='References'></a>References
[1] [Bordes et al., "Translating embeddings for modeling multi- relational data," in Adv. Neural Inf. Process. Syst., 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)


[2] [Wang, Z., Zhang, J., Feng, J., & Chen, Z. (2014). Knowledge Graph Embedding by Translating on Hyperplanes. Proceedings of the AAAI Conference on Artificial Intelligence, 28(1)](https://doi.org/10.1609/aaai.v28i1.8870)

[3][Ji et al. "Knowledge Graph Embedding via Dynamic Mapping Matrix", ACL-IJCNLP 2015](https://aclanthology.org/P15-1067) 


[4][Lin, Y., Liu, Z., Sun, M., Liu, Y., & Zhu, X. (2015). Learning Entity and Relation Embeddings for Knowledge Graph Completion. Proceedings of the AAAI Conference on Artificial Intelligence, 29(1). ](https://doi.org/10.1609/aaai.v29i1.9491)


[5][Ebisu et al. "TorusE: Knowledge Graph Embedding on a Lie Group", 2017](https://arxiv.org/abs/1711.05435)


[6][Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel. 2011. A three-way model for collective learning on multi-relational data. In Proceedings of the 28th International Conference on International Conference on Machine Learning (ICML'11). Omnipress, Madison, WI, USA, 809–816.
](https://doi.org/10.1609/aaai.v29i1.9491 )


[7][Yang et al. "Embedding Entities and Relations for Learning and Inference in Knowledge Bases", 2015](https://arxiv.org/abs/1412.6575)


[8][Trouillon et al. "Complex Embeddings for Simple Link Prediction", 2016](https://arxiv.org/abs/1606.06357)

[9][Liu et al. "Analogical Inference for Multi-Relational Embeddings", 2017](https://arxiv.org/abs/1705.02426)


[10][Nguyen et al. "A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network", 2018](http://dx.doi.org/10.18653/v1/N18-2053)


[11][Dettmers et al. "Convolutional 2D Knowledge Graph Embeddings", 2018](https://arxiv.org/abs/1707.01476)


[12][Nickel et al. "Holographic Embeddings of Knowledge Graphs", 2015](https://arxiv.org/abs/1510.04935)


[13][Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space", 2019](https://arxiv.org/abs/1902.10197)


[14][Balazevic et al. "TuckER: Tensor Factorization for Knowledge Graph Completion", 2019](http://dx.doi.org/10.18653/v1/D19-1522)

[16][Boschin, "TorchKGE: Knowledge Graph Embedding in Python and PyTorch", 2020](https://torchkge.readthedocs.io/en/latest/)

[17][Zhang et al. "Collaborative Knowledge Base Embedding for Recommender Systems", 2016](https://doi.org/10.1145/2939672.2939673)

[18][Ai et al. "Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation", 2018](http://dx.doi.org/10.3390/a11090137)

[19][Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation", 2019](http://dx.doi.org/10.1145/3292500.3330989)

[20][Xian et al. "Reinforcement Knowledge Graph Reasoning for Explainable Recommendation", 2019](https://doi.org/10.1145/3331184.3331203)

[21][Tai et al. "User-Centric Path Reasoning towards Explainable Recommendation", 2021](https://doi.org/10.1145/3404835.3462847)

[22][Xian et al. "CAFE: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation", 2020](http://dx.doi.org/10.1145/3340531.3412038)

[23][Geng et al. "Path Language Modeling over Knowledge Graphsfor Explainable Recommendation", 2022](https://doi.org/10.1145/3485447.3511937)

[24][Balloccu et al. "Faithful Path Language Modeling for Explainable Recommendation over Knowledge Graph", 2024](https://arxiv.org/abs/2310.16452)


## <a name='Contributing'></a>Contributing
This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know.

Please feel free to file issues and pull requests on the repo and we will address them as we can.
## <a name='Citations'></a>Citations
If you find this code useful in your work, please cite our papers:
```
@article{???,
  title={???},
  author={???},
  journal={arXiv ???},
  year={2024}
}
```
## <a name='License'></a>License
This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.
