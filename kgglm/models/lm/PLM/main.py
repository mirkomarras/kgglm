import argparse
import glob
import os
from datetime import datetime

import torch
import wandb
from datasets import load_from_disk
from transformers import (AutoConfig, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, PreTrainedTokenizerFast,
                          TrainingArguments, set_seed)

# Change to whatever model you want, this model will be trained and the embeddings used to initialize a gpt2 model.
from kgglm.models.kge.TransE.main import train as train_transe
from kgglm.models.kge.TransE.parser_transe import \
    parse_args as parse_args_transe
from kgglm.models.lm.PLM.lm_utils import (EmbeddingMapper,
                                           _initialise_type_masks,
                                           tokenize_augmented_kg)
from kgglm.models.lm.PLM.parser import parser_plm_args
from kgglm.models.lm.PLM.plmrec import PLMRec
from kgglm.models.lm.PLM.trainer import PathCLMTrainer
from kgglm.sampling.samplers.sampler import KGsampler
from kgglm.utils import SEED, check_dir, get_weight_dir


def update_config(config, tokenizer, args):
    # Assuming _initialise_type_masks is modified to work directly with tokenizer or adjust accordingly
    ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)
    config.update({
        'num_hops': args.n_hop,
        'sample_size_pretrain': args.sample_size,
        'sample_size_finetune': args.sample_size,
        'sample_size_hop': args.n_hop,
        'task': args.task,
        'train_batch_size': args.batch_size,
        'test_batch_size': args.test_batch_size,
        'ent_mask': ent_mask,
        'rel_mask': rel_mask,
        'token_id_to_token': token_id_to_token,
        # Any other configurations derived directly from args
    })
    return config

def run_kge_model(args):
    args_transe=parse_args_transe()
    args_transe.use_cuda='all'
    args_transe.dataset=args.dataset
    args_transe.embed_size=args.emb_size
    args_transe.epoch=args.num_epochs
    # todo: if embeddings are present in the directory don't train from scratch.
    print(f"[+] Training {args_transe.model_type} from Scratch")
    train_transe(args_transe)
    # Return oldest checkpoint saved
    transe_ckpts = glob.glob(os.path.join(args_transe.weight_dir_ckpt, '*.pth'))
    return min(transe_ckpts,key=os.path.getmtime)

def initialize_model_and_update_config(tokenizer,model,kg, args):
    model_name = model

    config_kwargs = {
        'vocab_size': len(tokenizer),
        'n_ctx': args.context_length,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    if args.pretrain_ckpt:
        print('Loading from checkpoint for resuming training:', args.pretrain_ckpt)
        model = PLMRec.from_pretrained(args.pretrain_ckpt,config=AutoConfig.from_pretrained(args.pretrain_ckpt, **config_kwargs))
    else:
        if 'plm-rec' in args.model:
            # Train a Translational model or whatever other kge method to get embeddings
            embedding_file=run_kge_model(args)
            embeds = torch.load(embedding_file,map_location=torch.device('cuda:0'))
            print('Using embeddings:', embedding_file.split('/')[-1])
            config_kwargs.update({
                'hidden_size': int(args.emb_size),
                'num_attention_heads': int(args.emb_size) // 10
            })
            config = AutoConfig.from_pretrained(args.model.split('@')[1], **config_kwargs)
            config = update_config(config, tokenizer, args)
            model = PLMRec(config).to(torch.device('cuda:0'))
            mapper = EmbeddingMapper(tokenizer, kg, embeds)
            mapper.init_with_embedding(model.transformer.wte.weight)
            print(f"Model {model_name} initialized with {embedding_file.split('/')[-1]} embeddings of size: {model.transformer.wte.weight.shape}")
        else:
            print("This train script must be use to train exclusively plm-rec@<model-base> models")
            exit(-1)
    return model

def prepare_training_arguments(args):
    trainer_logging_root = os.path.join(args.output_dir, args.exp_name, 'train_checkpoints')
    check_dir(trainer_logging_root)
    return TrainingArguments(
        output_dir=trainer_logging_root,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=args.validation_interval,
        logging_steps=min(args.logging_interval, args.validation_interval),
        learning_rate=2e-4,
        weight_decay=0.01,
        bf16=False,
        fp16=True,
        logging_first_step=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=250,
        save_steps=args.validation_interval,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='ndcg',
        greater_is_better=True,
        seed=SEED,  # Assuming SEED value
        report_to='wandb' if args.wandb else 'none',
    )

def train(args: argparse.Namespace, tokenizer, tokenized_dataset, kg):
    model = initialize_model_and_update_config(tokenizer, args.model, kg,args)
    tokenized_kg, _ = tokenize_augmented_kg(kg, tokenizer, use_token_ids=True)
    training_args = prepare_training_arguments(args)
    trainer = PathCLMTrainer(
        cmd_args=args,
        dataset_name=args.dataset,
        tokenized_kg=tokenized_kg,
        n_hop=args.n_hop,
        infer_batch_size=args.infer_batch_size,
        n_sequences_per_user=args.n_seq_infer,
        n_beams=args.n_beams,
        tokenizer=tokenizer,
        eval_device=args.eval_device,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        experiment_name=args.experiment_model_name,
        logit_processor_type=args.logit_processor_type,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()
    weight_path = get_weight_dir(args.experiment_model_name, args.dataset)
    trainer.save_model(weight_path)
    return model



if __name__ == "__main__":
    args=parser_plm_args()
    args.model = 'plm-rec@' + args.model
    set_seed(SEED)

    project_name = f'from_scratch_llm_v7@{args.dataset}'
    run_name=f"{args.exp_name}@{args.dataset}@{args.model}@{args.n_hop}@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = os.path.join(project_name, run_name)
    os.makedirs(log_dir, exist_ok=True)


    dataset_dir = os.path.join(args.data_dir, args.dataset)
    args.tokenizer_dir = './tokenizers'
    args.output_dir = log_dir 
    args.experiment_model_name = f"{args.task}@{args.dataset}@{args.model}@{args.sample_size}@{args.n_hop}@{args.logit_processor_type}"
    
    if args.wandb:

        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            name=run_name,
            # track hyperparameters and run metadata
            config=vars(args)
        )
    

    TOKENIZER_TYPE = "WordLevel"
    model_name = args.model
    dataset_name = args.dataset

    tokenizer_dir = os.path.join(args.tokenizer_dir, dataset_name)
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")

    dirpath = f'{args.data_dir}/{args.dataset}/preprocessed'
    data_dir_mapping = os.path.join(args.data_dir, f'{args.dataset}/preprocessed/mapping/')
    kg = KGsampler(args.dataset)
    sample_size = args.sample_size
    dataset_hop_size = args.n_hop
    TOKENIZED_DATASET_PATH = os.path.join(args.data_dir, f"{dataset_name}/{TOKENIZER_TYPE}/{args.task}_{sample_size}_{dataset_hop_size}_tokenized_dataset.hf")
    TOKEN_INDEX_PATH = os.path.join(dirpath, KGsampler.TOKEN_INDEX_FILE)
    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    if os.path.exists(TOKENIZED_DATASET_PATH) and os.path.exists(tokenizer_file):
        task = args.task
        tokenized_dataset = load_from_disk(TOKENIZED_DATASET_PATH)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)
    else:
        print("Tokenizer not found, run tokenization process before training")

    # Train the model
    if args.load_model:
        # Training arguments
        curr_sample_size = args.sample_size
        custom_name = f'clm-{args.task}-{args.dataset}-{args.model}-{curr_sample_size}-{args.n_hop}-{args.logit_processor_type}/checkpoint-{args.eval_ckpt_iter}'  # f"clm-from_scratch-{args.dataset}-{args.model}"
        model = AutoModelForCausalLM.from_pretrained(custom_name)
    else:
        model = train(args, tokenizer, tokenized_dataset, kg)