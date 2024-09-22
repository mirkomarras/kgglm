from typing import Dict

import numpy as np
import torch
import wandb
from datasets import Dataset
from tqdm import tqdm
from transformers import LogitsProcessorList, Trainer

from helper.evaluation.eval_metrics import evaluate_rec_quality
from helper.evaluation.eval_utils import (get_set, save_topks_items_results,
                                          save_topks_paths_results)
from helper.models.kge.utils import (get_kg_positives_and_tokens_ids_lp,
                                     get_set_lp, metrics_lp)
from helper.models.lm.KGGLM.decoding_constraints import ConstrainedLogitsProcessorLP, ConstrainedLogitsProcessorREC
from helper.models.lm.KGGLM.lm_utils import get_user_negatives_and_tokens_ids
from helper.models.lm.KGGLM.ranker import CumulativeSequenceScoreRanker, RankerLP
from helper.utils import get_dataset_id2eid


class PathPretrainTrainer(Trainer):
    def __init__(
            self,
            cmd_args=None,
            dataset_name=None,
            n_hop=3,
            n_epochs=20,
            infer_batch_size=1,
            n_sequences_per_user=10,
            n_sequences_lp=50,
            n_beams=30,
            n_beams_lp=50,
            tokenizer=None,
            eval_device='cpu',
            tokenized_kg=None,
            experiment_name=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.cmd_args = cmd_args
        self.model = kwargs['model']
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.custom_model_name = self.model.name_or_path.split("/")[-1]

        self.N_RET_SEQ = n_sequences_per_user
        self.N_RET_SEQ_LP= n_sequences_lp
        self.N_BEAMS = n_beams
        self.N_BEAMS_LP = n_beams_lp
        self.INFERENCE_BATCH_SIZE = infer_batch_size
        self.N_SEQUENCES_PER_USER = n_sequences_per_user

        self.n_hop = n_hop
        self.n_epochs=n_epochs
        self.eval_device = eval_device

        # Recommendation data
        self.test_set = get_set(dataset_name, set_str='test')
        uids = list(self.test_set.keys())
        self.user_negatives, self.user_negatives_token_ids = get_user_negatives_and_tokens_ids(dataset_name, tokenizer)
        self.token_id_to_uid_token_map = {tokenizer.convert_tokens_to_ids(f'U{uid}'): uid for uid in uids}
        init_condition_fn_rec = lambda uid: f"[BOS] U{uid} R-1"
        self.inference_paths_rec = {'uid': [init_condition_fn_rec(uid) for uid in uids]}
        self.SEQUENCE_LEN_REC = 2 * 3 + 2
        self.ranker_rec = CumulativeSequenceScoreRanker(tokenizer, user_negatives=self.user_negatives, K=10,
                                                        max_new_tokens=self.SEQUENCE_LEN_REC - len(
                                                            init_condition_fn_rec(0).split()))
        self.test_dataset_rec = Dataset.from_dict(self.inference_paths_rec)
        self.last_item_idx = max([int(id) for id in get_dataset_id2eid(dataset_name, 'product').values()])

        # Link Prediction Data
        self.SEQUENCE_LEN_LP = 3 + 1
        self.test_set_lp = get_set_lp(dataset_name,'test')
        heads_lp = [head for head, rel in self.test_set_lp.keys()]
        relations_lp = [rel for head, rel in self.test_set_lp.keys()]

        self.product_entities = [int(h) for h in get_dataset_id2eid(dataset_name, 'product').values()]
        self.all_entities, self.positive_triplets, self.positive_triplets_token_ids = get_kg_positives_and_tokens_ids_lp(dataset_name, tokenizer)

        init_condition_fn_lp = lambda head, rel: (f"[BOS] E{head} R{rel}" if head not in self.product_entities else f"[BOS] P{head} R{rel}")
        self.inference_paths_lp = {
            'eid_rid': [init_condition_fn_lp(head, rel) for head, rel in zip(heads_lp, relations_lp)]}
        self.ranker_lp = RankerLP(tokenizer, kg_positives=self.positive_triplets, K=10,
                                                       max_new_tokens=self.SEQUENCE_LEN_LP)
        self.test_dataset_lp = Dataset.from_dict(self.inference_paths_lp)
        print(f'Sequence length rec: {self.SEQUENCE_LEN_REC}, lp: {self.SEQUENCE_LEN_LP}')


        self.logits_processor_rec = LogitsProcessorList([
            ConstrainedLogitsProcessorREC(tokenized_kg=tokenized_kg,
                                force_token_map=self.user_negatives_token_ids,
                                tokenizer=tokenizer,
                                total_length=self.SEQUENCE_LEN_REC,
                                num_return_sequences=self.N_SEQUENCES_PER_USER,
                                id_to_uid_token_map=self.token_id_to_uid_token_map,
                                eos_token_ids=[
                                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                                )
        ])

        # Logit Processor Link Prediction
        self.logits_processor_lp = LogitsProcessorList([
            ConstrainedLogitsProcessorLP(tokenized_kg=tokenized_kg,
                                         positive_token_map=self.positive_triplets_token_ids,
                                         tokenizer=tokenizer,
                                         total_length=self.SEQUENCE_LEN_LP,
                                         num_return_sequences=self.N_SEQUENCES_PER_USER,
                                         eos_token_ids=[
                                             self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                                         )
        ])
    
    
    def __generate_topks_rec(self, model):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        batch_size = self.INFERENCE_BATCH_SIZE
        with tqdm(initial=0, desc="Generating topks", colour="green", total=len(self.user_negatives)) as pbar:
            for i in range(0, len(self.test_dataset_rec), batch_size):
                batch = self.test_dataset_rec[i:i + batch_size]
                inputs = self.tokenizer(batch["uid"], return_tensors='pt', add_special_tokens=False, ).to(
                    self.eval_device)

                outputs = model.generate(
                    **inputs,
                    max_length=self.SEQUENCE_LEN_REC,
                    min_length=self.SEQUENCE_LEN_REC,
                    num_return_sequences=self.N_RET_SEQ,
                    num_beams=self.N_BEAMS,
                    length_penalty=0.,
                    num_beam_groups=5,
                    diversity_penalty=0.3,
                    do_sample=False,
                    # top_p=0.4,
                    logits_processor=self.logits_processor_rec,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                self.ranker_rec.update_topk(outputs)
                pbar.update(batch_size)
        print("Average topk length:", sum(len(v) for v in self.ranker_rec.topk.values()) / max(len(self.ranker_rec.topk), 1))
        # print("Percentage of sequence that contain invalid item:", count/len(sorted_sequences))
        topks, topk_sequences = self.ranker_rec.topk, self.ranker_rec.topk_sequences
        save_topks_items_results(self.dataset_name, self.experiment_name + '_zeroshot_rec_'+str(self.n_epochs), topks, self.ranker_rec.K)
        save_topks_paths_results(self.dataset_name, self.experiment_name + '_zeroshot_rec_'+str(self.n_epochs), topk_sequences, self.ranker_rec.K)
        self.ranker_rec.reset_topks()

        return topks

    def __generate_topks_lp(self, model):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        batch_size = self.INFERENCE_BATCH_SIZE
        with tqdm(initial=0, desc="Generating topks", colour="green", total=len(self.test_set_lp)) as pbar:
            for i in range(0, len(self.test_set_lp), batch_size):
                batch = self.test_dataset_lp[i:i + batch_size]
                inputs = self.tokenizer(batch["eid_rid"], return_tensors='pt', add_special_tokens=False, ).to(self.eval_device)
                outputs = model.generate(
                    **inputs,
                    max_length=self.SEQUENCE_LEN_LP,
                    min_length=self.SEQUENCE_LEN_LP,
                    num_return_sequences=self.N_RET_SEQ_LP,
                    num_beams=self.N_BEAMS_LP,
                    length_penalty=0.,
                    num_beam_groups=5,
                    diversity_penalty=0.3,
                    do_sample=False,
                    logits_processor=self.logits_processor_lp,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                self.ranker_lp.update_topk(outputs)
                pbar.update(batch_size)
        print("Average topk length:", sum(len(v) for v in self.ranker_lp.topk.values()) / max(len(self.ranker_lp.topk), 1))
        # print("Percentage of sequence that contain invalid item:", count/len(sorted_sequences))
        topks, topk_sequences = self.ranker_lp.topk, self.ranker_lp.topk_sequences
        save_topks_items_results(self.dataset_name, self.experiment_name+ '_zeroshot_lp_'+str(self.n_epochs), topks, self.ranker_rec.K)
        save_topks_paths_results(self.dataset_name, self.experiment_name+ '_zeroshot_lp_'+str(self.n_epochs), topk_sequences, self.ranker_rec.K)
        self.ranker_lp.reset_topks()

        return topks

    def evaluate(self, model):
        self.callback_handler.on_epoch_end(self.args, self.state, self.control)  # call on_epoch_end callback
        # Generate paths for the test users
        # This heuristic assume that our scratch models use wordlevel and ft models use BPE, not ideal but for now is ok
        model.eval()
        model=torch.compile(model)
        topks_lp = self.__generate_topks_lp(model)
        metrics_lp_d = dict()

        avg_rec_quality_metrics = metrics_lp(self.test_set_lp, topks_lp)
        for k in avg_rec_quality_metrics:
            metrics_lp_d[f'eval_{k}'] = avg_rec_quality_metrics[k]

        topks_rec = self.__generate_topks_rec(model)

        metrics_ = dict()

        _, avg_rec_quality_metrics = evaluate_rec_quality(self.dataset_name, topks_rec, self.test_set,method_name='KGGLM')
        for k in avg_rec_quality_metrics:
            metrics_[f'eval_{k}'] = np.mean(avg_rec_quality_metrics[k])
        return metrics_

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):

        logs: Dict[str, float] = {}
        if self.control.should_log:
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

        metrics = None
        if self.control.should_evaluate and self.control.should_save:
            metrics = self.evaluate(model)
            logs.update(metrics)
            if self.cmd_args.wandb:
                wandb.log(logs)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that evaluation are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics[self.cmd_args.metric_for_best_model])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.cmd_args, self.state, self.control)

        # finish logging results
        if self.control.should_log:
            self.log(logs)

class PathFinetuneLinkPredictionTrainer(Trainer):
    def __init__(
            self,
            cmd_args=None,
            dataset_name=None,
            n_hop=3,
            n_epochs=20,
            infer_batch_size=1,
            n_sequences_lp=10,
            n_beams_lp=30,
            tokenizer=None,
            eval_device='cpu',
            tokenized_kg=None,
            experiment_name=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.cmd_args = cmd_args
        model = kwargs['model']
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.custom_model_name = model.name_or_path.split("/")[-1]

        self.N_RET_SEQ = n_sequences_lp
        self.N_BEAMS = n_beams_lp
        self.INFERENCE_BATCH_SIZE = infer_batch_size
        self.N_SEQUENCES_PER_USER = n_sequences_lp

        self.n_hop = n_hop
        self.n_epochs=n_epochs
        self.eval_device = eval_device

        # Link Prediction Data
        self.SEQUENCE_LEN_LP = 3 + 1
        self.test_set_lp = get_set_lp(dataset_name,'test')
        heads_lp = [head for head, rel in self.test_set_lp.keys()]
        relations_lp = [rel for head, rel in self.test_set_lp.keys()]

        self.product_entities = [int(h) for h in get_dataset_id2eid(dataset_name, 'product').values()]
        self.all_entities, self.positive_triplets, self.positive_triplets_token_ids = get_kg_positives_and_tokens_ids_lp(dataset_name, tokenizer)
        init_condition_fn_lp = lambda head, rel: (f"[BOS] E{head} R{rel}" if head not in self.product_entities else f"[BOS] P{head} R{rel}")
        self.inference_paths_lp = {
            'eid_rid': [init_condition_fn_lp(head, rel) for head, rel in zip(heads_lp, relations_lp)]}
        self.ranker_lp = RankerLP(tokenizer, kg_positives=self.positive_triplets, K=10,
                                                       max_new_tokens=self.SEQUENCE_LEN_LP)
        self.test_dataset_lp = Dataset.from_dict(self.inference_paths_lp)
        print(f'Sequence length lp: {self.SEQUENCE_LEN_LP}')

        print('Using: ', ConstrainedLogitsProcessorLP)
        logit_proc_kwargs = {}
        # Logit Processor Link Prediction
        self.logits_processor_lp = LogitsProcessorList([
            ConstrainedLogitsProcessorLP(tokenized_kg=tokenized_kg,
                                         positive_token_map=self.positive_triplets_token_ids,
                                         tokenizer=tokenizer,
                                         total_length=self.SEQUENCE_LEN_LP,
                                         num_return_sequences=self.N_SEQUENCES_PER_USER,
                                         eos_token_ids=[
                                             self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)],
                                         **logit_proc_kwargs
                                         )
        ])


    
    def __generate_topks_lp(self, model):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        batch_size = self.INFERENCE_BATCH_SIZE
        with tqdm(initial=0, desc="Generating topks", colour="green", total=len(self.test_set_lp)) as pbar:
            for i in range(0, len(self.test_set_lp), batch_size):
                batch = self.test_dataset_lp[i:i + batch_size]
                inputs = self.tokenizer(batch["eid_rid"], return_tensors='pt', add_special_tokens=False, ).to(
                    self.eval_device)

                outputs = model.generate(
                    **inputs,
                    max_length=self.SEQUENCE_LEN_LP,
                    min_length=self.SEQUENCE_LEN_LP,
                    num_return_sequences=self.N_RET_SEQ,
                    num_beams=self.N_BEAMS,
                    length_penalty=0.,
                    num_beam_groups=5,
                    diversity_penalty=0.3,
                    do_sample=False,
                    # top_p=0.4,
                    logits_processor=self.logits_processor_lp,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                self.ranker_lp.update_topk(outputs)
                pbar.update(batch_size)
        print("Average topk length:", sum(len(v) for v in self.ranker_lp.topk.values()) / max(len(self.ranker_lp.topk), 1))
        # print("Percentage of sequence that contain invalid item:", count/len(sorted_sequences))
        topks, topk_sequences = self.ranker_lp.topk, self.ranker_lp.topk_sequences
        save_topks_items_results(self.dataset_name, self.experiment_name+ '_lp_'+ str(self.n_epochs), topks, self.ranker_lp.K)
        save_topks_paths_results(self.dataset_name, self.experiment_name+ '_lp_'+str(self.n_epochs), topk_sequences, self.ranker_lp.K)
        self.ranker_lp.reset_topks()

        return topks

    def evaluate(self, model):
        # Generate paths for the test users
        # This heuristic assume that our scratch models use wordlevel and ft models use BPE, not ideal but for now is ok
        self.callback_handler.on_epoch_end(self.args,self.state,self.control) # call on_epoch_end callback

        topks_lp = self.__generate_topks_lp(model)
        metrics_lp_d = dict()

        avg_rec_quality_metrics = metrics_lp(self.test_set_lp, topks_lp)
        for k in avg_rec_quality_metrics:
            metrics_lp_d[f'eval_{k}'] = avg_rec_quality_metrics[k]

        return metrics_lp_d

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):

        logs: Dict[str, float] = {}
        if self.control.should_log:
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

        metrics = None
        if self.control.should_evaluate and self.control.should_save:
            metrics = self.evaluate(model)
            logs.update(metrics)
            if self.cmd_args.wandb:
                wandb.log(logs)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that evaluation are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics[self.cmd_args.metric_for_best_model])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.cmd_args, self.state, self.control)

        # finish logging results
        if self.control.should_log:
            self.log(logs)

class PathFinetuneExplainableRecTrainer(Trainer):
    def __init__(
            self,
            cmd_args=None,
            dataset_name=None,
            n_hop=3,
            n_epochs=20,
            infer_batch_size=1,
            n_sequences_per_user=10,
            n_beams=30,
            tokenizer=None,
            eval_device='cpu',
            tokenized_kg=None,
            experiment_name=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.cmd_args = cmd_args
        model = kwargs['model']
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.custom_model_name = model.name_or_path.split("/")[-1]
        self.n_epochs=n_epochs
        self.N_RET_SEQ = n_sequences_per_user
        self.N_BEAMS = n_beams
        self.INFERENCE_BATCH_SIZE = infer_batch_size
        self.N_SEQUENCES_PER_USER = n_sequences_per_user

        self.n_hop = n_hop
        self.eval_device = eval_device

        # Recommendation data
        self.test_set = get_set(dataset_name, set_str='test')
        uids = list(self.test_set.keys())
        self.user_negatives, self.user_negatives_token_ids = get_user_negatives_and_tokens_ids(dataset_name, tokenizer)
        self.token_id_to_uid_token_map = {tokenizer.convert_tokens_to_ids(f'U{uid}'): uid for uid in uids}
        init_condition_fn_rec = lambda uid: f"[BOS] U{uid} R-1"
        self.inference_paths_rec = {'uid': [init_condition_fn_rec(uid) for uid in uids]}
        self.SEQUENCE_LEN_REC = 2 * 3 + 2
        self.ranker_rec = CumulativeSequenceScoreRanker(tokenizer, user_negatives=self.user_negatives, K=10,
                                                        max_new_tokens=self.SEQUENCE_LEN_REC - len(
                                                            init_condition_fn_rec(0).split()))
        self.test_dataset_rec = Dataset.from_dict(self.inference_paths_rec)
        self.last_item_idx = max([int(id) for id in get_dataset_id2eid(dataset_name, 'product').values()])

        print(f'Sequence length rec: {self.SEQUENCE_LEN_REC}')



        self.logits_processor_rec = LogitsProcessorList([
            ConstrainedLogitsProcessorREC(tokenized_kg=tokenized_kg,
                                force_token_map=self.user_negatives_token_ids,
                                tokenizer=tokenizer,
                                total_length=self.SEQUENCE_LEN_REC,
                                num_return_sequences=self.N_SEQUENCES_PER_USER,
                                id_to_uid_token_map=self.token_id_to_uid_token_map,
                                eos_token_ids=[
                                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                                )
        ])


    
    def __generate_topks_rec(self, model):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        batch_size = self.INFERENCE_BATCH_SIZE
        with tqdm(initial=0, desc="Generating topks", colour="green", total=len(self.user_negatives)) as pbar:
            for i in range(0, len(self.test_dataset_rec), batch_size):
                batch = self.test_dataset_rec[i:i + batch_size]
                inputs = self.tokenizer(batch["uid"], return_tensors='pt', add_special_tokens=False, ).to(
                    self.eval_device)

                outputs = model.generate(
                    **inputs,
                    max_length=self.SEQUENCE_LEN_REC,
                    min_length=self.SEQUENCE_LEN_REC,
                    num_return_sequences=self.N_RET_SEQ,
                    num_beams=self.N_BEAMS,
                    length_penalty=0.,
                    num_beam_groups=5,
                    diversity_penalty=0.3,
                    do_sample=False,
                    # top_p=0.4,
                    logits_processor=self.logits_processor_rec,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                self.ranker_rec.update_topk(outputs)
                pbar.update(batch_size)
        print("Average topk length:", sum(len(v) for v in self.ranker_rec.topk.values()) / max(len(self.ranker_rec.topk), 1))
        # print("Percentage of sequence that contain invalid item:", count/len(sorted_sequences))
        topks, topk_sequences = self.ranker_rec.topk, self.ranker_rec.topk_sequences
        save_topks_items_results(self.dataset_name, self.experiment_name + '_rec_'+str(self.n_epochs), topks, self.ranker_rec.K)
        save_topks_paths_results(self.dataset_name, self.experiment_name + '_rec_'+str(self.n_epochs), topk_sequences, self.ranker_rec.K)
        self.ranker_rec.reset_topks()

        return topks


    def evaluate(self, model):
        self.callback_handler.on_epoch_end(self.args, self.state, self.control)  # call on_epoch_end callback

        # Generate paths for the test users
        topks_rec = self.__generate_topks_rec(model)

        metrics_ = dict()

        _, avg_rec_quality_metrics = evaluate_rec_quality(self.dataset_name, topks_rec, self.test_set)
        for k in avg_rec_quality_metrics:
            metrics_[f'eval_{k}'] = np.mean(avg_rec_quality_metrics[k])

        return metrics_

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):

        logs: Dict[str, float] = {}
        if self.control.should_log:
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

        metrics = None
        if self.control.should_evaluate and self.control.should_save:
            metrics = self.evaluate(model)
            logs.update(metrics)
            if self.cmd_args.wandb:
                wandb.log(logs)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that evaluation are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics[self.cmd_args.metric_for_best_model])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.cmd_args, self.state, self.control)

        # finish logging results
        if self.control.should_log:
            self.log(logs)


