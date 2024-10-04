from typing import Dict, Tuple

from tqdm import tqdm

from kgglm.evaluation.beyond_accuracy_metrics import (COVERAGE, DIVERSITY,
                                                       NOVELTY, PFAIRNESS,
                                                       SERENDIPITY, coverage,
                                                       diversity_at_k,
                                                       get_item_count,
                                                       get_item_genre,
                                                       get_item_pop,
                                                       novelty_at_k,
                                                       serendipity_at_k)
from kgglm.evaluation.eval_utils import (REC_QUALITY_METRICS_TOPK,
                                          compute_mostpop_topk,
                                          get_precomputed_topks)
from kgglm.evaluation.utility_metrics import *


def print_rec_quality_metrics(avg_rec_quality_metrics: Dict[str, float], method='inline'):
    """
    args:
        avg_rec_quality_metrics: a dictionary containing the average value of each metric
    """
    print(', '.join([f'{metric}: {round(value, 2)}' for metric, value in avg_rec_quality_metrics.items()]))


def evaluate_rec_quality_from_results(dataset_name: str, model_name: str, test_labels: Dict[int, List[int]],
                                      k: int = 10, metrics: List[str] = REC_QUALITY_METRICS_TOPK) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    This function computes all the recommendation quality metrics for a given set of topk items that are already computed
    and stored in the results folder.
    """
    topks = get_precomputed_topks(dataset_name, model_name)
    # TOPK size is fixed to 10
    return evaluate_rec_quality(dataset_name, topks, test_labels, k, metrics)


def evaluate_rec_quality(dataset_name: str, topk_items: Dict[int, List[int]], test_labels: Dict[int, List[int]],
                         k: int = 10, method_name=None, metrics: List[str] = REC_QUALITY_METRICS_TOPK) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    This function computes all the recommendation quality metrics for a given set of topk items, please note that the topk items and test set are
    expressed using the original ids of the dataset (e.g. the ids of the movies in the MovieLens dataset).
    """
    rec_quality_metrics = {metric: list() for metric in metrics}
    recommended_items_all_user_set = set()

    n_items_in_catalog = get_item_count(dataset_name)  # Needed for coverage
    pid2popularity = get_item_pop(dataset_name)  # Needed for novelty
    pid2genre = get_item_genre(dataset_name)  # Needed for diversity
    mostpop_topk = compute_mostpop_topk(dataset_name, k)  # Needed for serendipity

    topk_sizes = []
    # Evaluate recommendation quality for users' topk
    for uid, topk in tqdm(
        topk_items.items(), desc=f"Evaluating rec quality for {method_name}", total=len(topk_items.keys()), position=1, leave=False
    ):
        hits = []
        for pid in topk[:k]:
            hits.append(1 if pid in test_labels[uid] else 0)

        # If the model has predicted less than 10 items pad with zeros
        while len(hits) < k:
            hits.append(0)
        for metric in metrics:
            if len(topk) == 0:
                metric_value = 0.0
            else:
                if metric == NDCG:
                    metric_value = ndcg_at_k(hits, k)
                if metric == MRR:
                    metric_value = mmr_at_k(hits, k)
                if metric == PRECISION:
                    metric_value = precision_at_k(hits, k)
                if metric == RECALL:
                    test_set_len = max(max(1, len(topk)), len(test_labels[uid]))
                    metric_value = recall_at_k(hits, k, test_set_len)
                if metric == SERENDIPITY and mostpop_topk is not None:
                    metric_value = serendipity_at_k(topk, mostpop_topk[uid], k)
                if metric == DIVERSITY and pid2genre is not None:
                    metric_value = diversity_at_k(topk, pid2genre)
                if metric == NOVELTY and pid2popularity is not None:
                    metric_value = novelty_at_k(topk, pid2popularity)
                if metric == PFAIRNESS:
                    continue  # Skip for now
            rec_quality_metrics[metric].append(metric_value)

        # For coverage
        recommended_items_all_user_set.update(set(topk))
        topk_sizes.append(len(topk))

    # Compute average values for evaluation
    avg_rec_quality_metrics = {metric: np.mean(values) for metric, values in rec_quality_metrics.items()}
    avg_rec_quality_metrics[COVERAGE] = coverage(recommended_items_all_user_set, n_items_in_catalog)

    # Print results
    print(f'Number of users: {len(test_labels.keys())}, average topk size: {np.array(topk_sizes).mean():.2f}')
    print_rec_quality_metrics(avg_rec_quality_metrics)
    # print(generate_latex_row(args.model, avg_rec_quality_metrics, "rec"))
    # Save as csv if specified
    return rec_quality_metrics, avg_rec_quality_metrics




