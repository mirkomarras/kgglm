# KGGLM: A Generative Language Model for Generalizable Knowledge Graph Representation in Recommendation


**Note:** all experiments have been run with fixed seed in order to ease reproducibility of the results.

<!-- vscode-markdown-toc -->
* [Requirements](#Requirements)
* [Usage](#Usage)
	* [Run the Experiments](#RuntheExperiments)
	* [Reproducibility](#Reproducibility)
* [Datasets](#Datasets)
* [Results](#Results)
	* [Recommendation](#Recommendation)
	* [Knowledge Completion](#KnowledgeCompletion)
* [Contributing](#Contributing)
* [License](#License)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='Requirements'></a>Requirements
- Python 3.9.13
 

It is possible to download the weights and paths for KGGLM from *(LINK AVAILABLE UPON ACCEPTANCE)*

*Make sure to download the `data/` folder present in onedrive and copy it into the `fix/` folder before proceeding.*

## <a name='Usage'></a>Usage
**Steps.**
1. Clone of the repository by running `git clone --recurse-submodules https://github.com/mirkomarras/lm-vs-gnn-embeddings.git`
2. Execute `chmod +x fix/fix.sh && ./fix/fix.sh`
3. Install the required packages with `pip install -r requirements.txt`
4. Install helper with `pip install helper/.`
5. Run `CUDA_DEVICE=0 && ./run_kge_experiments_recommendation.sh CUDA_DEVICE` to train and view the results for the baselines on Recommendation
6. Run `CUDA_DEVICE=0 && ./run_kge_evaluation_recommendation.sh CUDA_DEVICE` to view the results on the best embeddings for the baselines on Recommendation
7. Run `CUDA_DEVICE=0 && ./run_kge_experiments_linkprediction.sh CUDA_DEVICE` to train and view the results for the baselines on Link Prediction
8. Run `CUDA_DEVICE=0 && ./run_kge_evaluation_linkprediction.sh CUDA_DEVICE` to view the results on the best embeddings for the baselines on Link Prediction
9. Run `CUDA_DEVICE=0 && ./run_kgglm_experiments.sh CUDA_DEVICE` to view the results on the best embeddings for KKGLM on Recommendation and Link Prediction


 
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
export CUDA_VISIBLE_DEVICES=$GPU && python helper/pathlm/models/kge_rec/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_$dataset.txt && end=$(date +%s) && runtime=$((end-start)) && echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds\n" >> elapsed_training.txt
```

### <a name='Reproducibility'></a>Reproducibility 

The reproducibility of the experiments can be done by running `./run_kge_evaluations_recommendation.sh CUDA_DEVICE` or `./run_kge_evaluations_linkprediction.sh CUDA_DEVICE` where the models are executed sequentially with the default embeddings the ones that results to the best performance on the datasets.

First of all it is necessary to download and copy the folders "Best Checkpoint for Reproducibility" and "kgglm Weights" from the link above (*LINK AVAILABLE UPON ACCEPTANCE*) to download the best weights, then you can run `./setup_reproducibility.sh` to automatically copy the weights in the correct folders (after you've downloaded the folder and moved it on the root). Finally, execute `./run_kge_evaluations_recommendation.sh CUDA_DEVICE` or `./run_kgglm_experiments.sh CUDA_DEVICE` to test the models on their best weights. 

By executing it, you will obtain the same results as in the paper.

For Example, for TransE on ml1m and lfm1m for Recommendation:
```sh
# On ml1m
python helper/pathlm/models/kge_rec/TransE/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TransE_dataset_ml1m_ndcg_0.28_mrr_0.23_prec_0.1_rec_0.03_ser_0.33_div_0.41_nov_0.93_cov_0.04_epoch_16_e100_bs64_lr0.0001.pth

# On lfm1m
python helper/pathlm/models/kge_rec/TransE/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TransE_dataset_lfm1m_ndcg_0.12_mrr_0.1_prec_0.03_rec_0.01_ser_0.59_div_0.54_nov_0.85_cov_0.0_epoch_28_e100_bs64_lr0.001.pth

```
>*Remember to change the embed_size parameter when you change the checkpoint for a model. In general, to test other weights, you have to move the weight ".pth" inside helper/pathlm/models/kge_rec/TransE/weight_dir_ckpt if the model to test is TransE for example. You can evaluate on link prediction task by changing the checkpoint and setting --lp True*


## <a name='Datasets'></a>Datasets

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


## <a name='Results'></a>Results
### <a name='Recommendation'></a>Recommendation
**ML1M**
|          | NDCG | MRR  |  Parameters: [Epoch, EmbSize, BatchSize, lr] |
|----------|------|------|-----------------------------------------------|
| TransE   | 0.28 |  0.23  |         [16,100,64,0.0001]            |
| TransH    |  0.26 | 0.20|        [10,100,64,0.0001]              |
| TransD   |  0.23  |  0.18  |        [27,100,256,0.0001]          |
| TransR   |  0.16  |  0.12  |       [2,100,256,0.001]           |
| TorusE  |  0.26  |   0.21   |        [0,100,64,0.0001]                    |
| RotatE  |  0.20  |  0.15 |        [23,100,256,0.0001]       |
| RESCAL  |  0.26  |  0.20  |         [3,100,64,0.0001]               |
| DistMult |  0.28  |  0.22   |        [22,100,64,0.0001]                  |
| ComplEX | 0.26   | 0.21  |  [2,100,64,0.0001]  | 
| TuckER  |  0.27  | 0.22    |         [19,100,256,0.0001]                 |
| Analogy  | 0.26  |  0.21    |[4,100,64,0.0001]         |
| HolE    | 0.24  |  0.19    |      [11,100,64,0.0001]                |
| ConvE   |  0.27  |  0.22   |       [2,200,256,0.0001]                 |
| MostPopular | 0.26| 0.22|  | 
| BPRMF   | 0.29|0.23 | [300, 64,1024,0.0001] | 
| CKE | 0.30 | 0.23 | [120, 64, 1024,0.001] | 
| CFKG   |  0.27  |  0.21    |   [500, 64, 1024, 0.0001]     |
| KGAT   |  0.31  |  0.24   |  [500,64,1024,0.0001]       |
| PGPR   |  0.28 |  0.21   |    [50,512 , 64, 0.0001]     |
| UCPR   |  0.26  |  0.19  |    [40, 64,128,0.0000007]     |
| CAFE   |  0.21  |  0.15  |    [20,200 , 64, 0.1]     |
| PLM   |  0.27  |  0.18    |  [20, 100, 256, 0.0002]       |
| KGGLM (gen-only)   |  0.11  |  0.07 |      [3,768, 256, 0.0002]     |
| KGGLM (gen+spec) |  0.41  |  0.31  |        [2,768, 256, 0.0002]  |






**LFM1M**


|          | NDCG | MRR  |  Parameters: [Epoch, EmbSize, BatchSize, lr] |
|----------|------|------|--------------------------------------------|
| TransE  | 0.12 | 0.10    |    [28,100,64,0.001]                  |
| TransH  |   0.18 | 0.14 |    [22,100,64,0.001]                    |
| TransD  |  0.17 |  0.13  |         [29,100,256,0.0001]               |
| TransR  |   0.15 |  0.11  |            [29,100,256,0.001]                  |
| TorusE  | 0.18   |  0.14  |         [28,100,256,0.0001]            |
| RotatE  | 0.25  |  0.20 |        [29,100,256,0.001]            |
| RESCAL | 0.24  | 0.19    |      [26,100,256,0.001]                     |
| DistMult | 0.30  |   0.25  |      [10,100,256,0.001]                |
| ComplEX |   0.28 | 0.23    |          [29,100,64,0.0001]   |
| TuckER  |   0.12 |  0.10  |         [4,64,64,0.0001]              |
| Analogy |  0.33   |  0.28  |  [27,100,128,0.0001]                                   |
| HolE    |  0.19  | 0.14   |             [29,100,64,0.0001]                 |
| ConvE   |  0.13  |  0.11   |   [2,200,64,0.0001]            |
| MostPopular | 0.12| 0.10| | 
| BPRMF   |0.08 |0.06 | [300, 64,1024,0.0001]| 
| CKE   |  0.33  |  0.27  |  [120, 64, 1024,0.001]       |
| CFKG   |  0.13  |  0.10  |   [500, 64, 1024, 0.0001]                |
| KGAT   |  0.30  |  0.24  |   [120, 64, 1024, 0.001]                 |
| PGPR   |  0.18 |  0.14  |    [50, 512 , 64, 0.0001]               |
| UCPR   |  0.32 |  0.26  |    [40, 64,128,0.0000007]          |
| CAFE   |  0.14  |  0.09  |      [20, 200, 64, 0.1]          |
| PLM   |  0.28  |  0.19  |  [20, 100, 256, 0.0002]               |
| KGGLM (gen-only)    |  0.27  |  0.20 |  [3,768, 256, 0.0002]               |
| KGGLM (gen+spec)  |  0.53  |  0.45 |  [2,768, 256, 0.0002]            |

### <a name='KnowledgeCompletion'></a>Knowledge Completion

**ml1m**
|           | MRR | Hits@1 |  Parameters: [Epoch, EmbSize, BatchSize, lr] |
|----------|------|----|----------------------------------------------|
| TransE |   0.19  |   0.14   |          [23, 100, 64, 0.0001]               |
| TransH  |  0.19   |    0.15    |         [24 ,100 ,64 ,0.0001]             |
| TransD |   0.16  |  0.07      |           [29, 100, 64, 0.0001]              |
| TransR  |   0.13  |  0.06      |           [29 ,64, 128, 0.01]            |
| TorusE |    0.20 |   0.15     |            [17, 100, 64, 0.0001]               |
| RotatE |  0.22   |    0.13   |        [12 ,64, 128, 0.01, 0]               |
| RESCAL  |   0.20  |     0.16   |           [3, 100, 64, 0.0001]              |
| DistMult  |  0.19   |   0.11     |        [29, 100, 64, 0.0001]                  |
| ComplEX  |     0.23 |     0.15   |       [29, 100, 64, 0.0001]                |
| TuckER   |   0.10  |      0.04  |           [26, 100, 256, 0.0001]              |
| Analogy  |   0.20  |    0.13    |        [18, 100, 64, 0.0001]              |
| HolE  | 0.14    |  0.10      |         [24, 100, 64, 0.0001]                |
| ConvE |   0.10  | 0.04       |       [1 ,200 ,64, 0.0001]                |
| KGGLM (gen-only) |   0.38     | 0.29     |    [3,768, 256, 0.0002]              |
| KGGLM (gen+spec)  |   0.39   | 0.30   |    [2,768, 256, 0.0002]           |

**lfm1m**

|                | MRR     | Hits@1        | Parameters: [Epoch, EmbSize, BatchSize, lr] |
|----------|------|-----|---------------------------------------------|
| TransE     |   0.76  |    0.66     |       [21, 100, 64, 0.0001]             |
| TransH  |   0.76  |   0.66         |           [21, 100, 64, 0.0001]         |
| TransD     |  0.31   | 0.01         |           [4, 100, 64, 0.0001]          |
| TransR  | 0.35    |    0.19        |            [18 ,64, 128, 0.01]          |
| TorusE   | 0.79    |    0.69       |          [7, 100, 64, 0.0001]           |
| RotatE    |  0.07   |    0.02      |      [0, 64, 128, 0.01]                 |
| RESCAL     |  0.29   |   0.21          |        [3, 100, 64, 0.0001]               |
| DistMult    |    0.17 |    0.07       |       [14 ,100, 64, 0.0001]               |
| ComplEX    |  0.68   |    0.58     |       [2, 100, 64, 0.0001]                 |
| TuckER  |  0.76   |    0.66        |       [15, 100, 64, 0.0001]                 |
| Analogy   |    0.50 |    0.37    |      [3, 100, 64, 0.0001]              |
| HolE       |  0.33   | 0.23        |       [8, 100, 64, 0.0001]             |
| ConvE    | 0.76    |    0.67        |      [11,200 ,64, 0.0001]                  |
| KGGLM (gen-only)  |  0.26 |   0.20      |    [3,768, 256, 0.0002]  |
| KGGLM (gen+spec)    |  0.09|   0.04    |       [2,768, 256, 0.0002]            |


## <a name='Contributing'></a>Contributing
This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know.

Please feel free to file issues and pull requests on the repo and we will address them as we can.
## <a name='License'></a>License
This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.
